####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'age'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_1: var_15, var_8: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'Jane'
    var_20 = 25
    var_21 = var_18.serialize(var_13)
    var_22 = module_0.Field()
    var_23 = module_0.Field()
    var_24 = 'email'
    var_25 = {var_1: var_22, var_8: var_23, var_24: var_22}
    var_26 = module_1.Schema(var_25)
    var_27 = 'Bob'
    var_28 = 35
    var_29 = {var_1: var_27, var_8: var_28}
    var_30 = var_26.serialize(var_29)
    var_31 = module_0.Field()
    var_32 = module_0.Field()
    var_33 = {var_1: var_31, var_8: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = 'Alice'
    var_36 = var_34.serialize(var_29)
    var_37 = 'message'
    var_38 = 'hello'
    var_39 = {var_37: var_38}
    var_40 = var_34.serialize(var_39)
    var_41 = {}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_1: var_11}
    var_44 = var_42.serialize(var_43)
    var_45 = 'data'
    var_46 = 'value'
    var_47 = {var_45: var_46}
    var_48 = var_42.serialize(var_47)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'key2'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'value1'
    var_12 = 'value2'
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_1: var_15, var_8: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = var_18.serialize(var_13)
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_1: var_20, var_8: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_1: var_11}
    var_25 = var_23.serialize(var_24)
    var_26 = module_0.Field()
    var_27 = {var_1: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'extra_key'
    var_30 = 'extra_value'
    var_31 = {var_1: var_11, var_29: var_30}
    var_32 = var_28.serialize(var_31)
    var_33 = {var_1: var_26}
    var_34 = module_1.Schema(var_33)
    var_35 = {var_1: var_11}
    var_36 = var_34.serialize(var_35)
    var_37 = module_0.Field()
    var_38 = module_0.Field()
    var_39 = {var_1: var_37, var_8: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = var_40.serialize(var_35)
    var_42 = module_0.Field()
    var_43 = module_0.Field()
    var_44 = {var_1: var_42, var_8: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {var_1: var_11}
    var_47 = var_45.serialize(var_46)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'age'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = {var_1: var_6, var_8: var_7}
    var_16 = module_1.Schema(var_15)
    var_17 = 'Alice'
    var_18 = 25
    var_19 = 'Bob'
    var_20 = {var_1: var_19}
    var_21 = var_16.serialize(var_20)
    var_22 = True
    var_23 = module_0.Field(read_only=var_22)
    var_24 = module_0.Field()
    var_25 = {var_1: var_23, var_8: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'Charlie'
    var_28 = 35
    var_29 = {var_1: var_27, var_8: var_28}
    var_30 = var_26.serialize(var_29)
    var_31 = {}
    var_32 = module_1.Schema(var_31)
    var_33 = 'Dave'
    var_34 = 40
    var_35 = {var_1: var_33, var_8: var_34}
    var_36 = var_32.serialize(var_35)
    var_37 = module_0.Field()
    var_38 = module_0.Field()
    var_39 = {var_1: var_37, var_8: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = var_40.serialize(var_35)
    var_42 = 'status'
    var_43 = module_0.Field()
    var_44 = 'test'
    var_45 = 'active'
    var_46 = {var_1: var_44, var_42: var_45}
    var_47 = var_40.serialize(var_46)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = None
    var_19 = var_6.validate(var_18)
    var_20 = module_0.Definitions()
    var_21 = 'id'
    var_22 = 'value'
    var_23 = module_1.Field()
    var_24 = module_1.Field()
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'AnotherSchema'
    var_28 = module_0.Reference(var_27, var_20)
    var_29 = {var_21: var_10, var_22: var_7}
    var_30 = var_28.validate(var_29)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = module_1.Field()
    var_15 = 'key1'
    var_16 = {var_15: var_14}
    var_17 = module_0.Schema(var_16)
    var_18 = 'value1'
    var_19 = {var_15: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = {}
    var_22 = module_0.Schema(var_21)
    var_23 = 1
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = module_1.Field(default=var_25)
    var_28 = module_1.Field()
    var_29 = 'field1'
    var_30 = 'field2'
    var_31 = {var_29: var_27, var_30: var_28}
    var_32 = module_0.Schema(var_31)
    var_33 = 'field1'
    var_34 = 'value1'
    var_35 = {var_33: var_34}
    var_36 = var_32.validate(var_35)
    var_37 = 'default_value'
    var_38 = module_1.Field(default=var_37)
    var_39 = {var_29: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = module_1.Field(read_only=var_34)
    var_44 = module_1.Field()
    var_45 = {var_29: var_43, var_30: var_44}
    var_46 = module_0.Schema(var_45)
    var_47 = 'ignored'
    var_48 = 'value2'
    var_49 = {var_29: var_47, var_30: var_48}
    var_50 = var_46.validate(var_49)
    var_51 = module_1.Field()
    var_52 = {var_15: var_51}
    var_53 = module_0.Schema(var_52)
    var_54 = (var_15, var_18)
    var_55 = [var_54]
    var_56 = module_1.Integer()
    var_57 = {var_29: var_56}
    var_58 = module_0.Schema(var_57)
    var_59 = 'field1'
    var_60 = 'not_an_integer'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = module_1.Field()
    var_64 = module_1.Field()
    var_65 = {var_29: var_63, var_30: var_64}
    var_66 = module_0.Schema(var_65)
    var_67 = {var_29: var_18, var_30: var_48}
    var_68 = var_66.validate(var_67)



# Parsed testcases at query #6
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
    var_6 = 'id'
    var_7 = module_0.String()
    var_8 = True
    var_9 = module_0.Integer()
    var_10 = {var_0: var_7, var_6: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'active'
    var_13 = module_0.String()
    var_14 = module_0.Boolean()
    var_15 = {var_0: var_13, var_12: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = module_0.String()
    var_18 = module_0.Integer()
    var_19 = False
    var_20 = module_0.Boolean()
    var_21 = {var_0: var_17, var_6: var_18, var_12: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = {}
    var_24 = module_1.Schema(var_23)
    var_25 = module_1.Schema(var_4)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    var_8 = 'id'
    var_9 = module_0.Field()
    var_10 = {var_8: var_7, var_0: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'default_value'
    var_13 = module_0.Field(default=var_12)
    var_14 = 'status'
    var_15 = module_0.Field()
    var_16 = {var_14: var_13, var_0: var_15}
    var_17 = module_1.Schema(var_16)
    var_18 = 'email'
    var_19 = module_0.Field(read_only=var_6)
    var_20 = 'active'
    var_21 = module_0.Field(default=var_20)
    var_22 = module_0.Field()
    var_23 = module_0.Field()
    var_24 = {var_8: var_19, var_14: var_21, var_0: var_22, var_18: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = {}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = module_1.Schema(var_28)
    var_30 = module_0.Field(read_only=var_6)
    var_31 = 'pending'
    var_32 = module_0.Field(default=var_31)
    var_33 = {var_8: var_30, var_14: var_32}
    var_34 = module_1.Schema(var_33)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = {}
    var_17 = var_15.validate(var_16)
    var_18 = {}
    var_19 = module_0.Schema(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = module_1.Field()
    var_25 = 'name'
    var_26 = {var_25: var_24}
    var_27 = module_0.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = module_1.Field()
    var_31 = {var_25: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'test'
    var_34 = {var_25: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_1.Field(read_only=var_29)
    var_37 = 'id'
    var_38 = {var_37: var_36}
    var_39 = module_0.Schema(var_38)
    var_40 = '123'
    var_41 = {var_37: var_40}
    var_42 = var_39.validate(var_41)
    var_43 = 'default_value'
    var_44 = module_1.Field(default=var_43)
    var_45 = 'status'
    var_46 = {var_45: var_44}
    var_47 = module_0.Schema(var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)
    var_50 = module_1.Integer()
    var_51 = 'age'
    var_52 = {var_51: var_50}
    var_53 = module_0.Schema(var_52)
    var_54 = 'age'
    var_55 = 'not_an_int'
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = module_1.String()
    var_59 = module_1.Integer()
    var_60 = {var_25: var_58, var_51: var_59}
    var_61 = module_0.Schema(var_60)
    var_62 = 'John'
    var_63 = 30
    var_64 = {var_25: var_62, var_51: var_63}
    var_65 = var_61.validate(var_64)
    var_66 = module_1.String()
    var_67 = {var_25: var_66}
    var_68 = module_0.Schema(var_67)
    var_69 = 'extra'
    var_70 = 'field'
    var_71 = {var_25: var_62, var_69: var_70}
    var_72 = var_68.validate(var_71)
    var_73 = {var_25: var_33}
    var_74 = module_1.String()
    var_75 = {var_25: var_74}
    var_76 = module_0.Schema(var_75)
    var_77 = module_1.String()
    var_78 = module_1.Integer()
    var_79 = {var_25: var_77, var_51: var_78}
    var_80 = module_0.Schema(var_79)
    var_81 = {}
    var_82 = var_80.validate(var_81)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Reference(var_5, var_0)
    var_19 = 'invalid'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Reference(var_5, var_0)
    var_22 = 'id'
    var_23 = 'value'
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.Schema(var_26)
    var_28 = 'ComplexSchema'
    var_29 = module_0.Reference(var_28, var_0)
    var_30 = {var_22: var_10, var_23: var_7}
    var_31 = var_29.validate(var_30)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = 'TestSchema'
    var_4 = True
    var_5 = module_0.Reference(var_3, var_0)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = False
    var_9 = module_0.Reference(var_3, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_1.Field()
    var_13 = 'name'
    var_14 = {var_13: var_12}
    var_15 = module_0.Schema(var_14)
    var_16 = 'UserSchema'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'John'
    var_19 = {var_13: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = module_0.Reference(var_16, var_0)
    var_22 = None
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Definitions()
    var_25 = {}
    var_26 = module_0.Schema(var_25)
    var_27 = 'MyDef'
    var_28 = module_0.Reference(var_27, var_24)
    var_29 = module_1.Field()
    var_30 = 'id'
    var_31 = 'data'
    var_32 = {var_30: var_29, var_31: var_29}
    var_33 = module_0.Schema(var_32)
    var_34 = 'ComplexSchema'
    var_35 = module_0.Reference(var_34, var_0)
    var_36 = 'test'
    var_37 = {var_30: var_4, var_31: var_36}
    var_38 = var_35.validate(var_37)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = 'age'
    var_15 = module_1.Field()
    var_16 = module_1.Field()
    var_17 = {var_11: var_15, var_14: var_16}
    var_18 = module_0.Schema(var_17)
    var_19 = 'Person'
    var_20 = module_0.Reference(var_19, var_13)
    var_21 = 'John'
    var_22 = 30
    var_23 = {var_11: var_21, var_14: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = module_0.Definitions()
    var_26 = module_1.Field()
    var_27 = {var_11: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'RequiredSchema'
    var_30 = module_0.Reference(var_29, var_25)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Definitions()
    var_34 = 'id'
    var_35 = module_1.Field()
    var_36 = {var_34: var_35}
    var_37 = module_0.Schema(var_36)
    var_38 = 'data'
    var_39 = {var_38: var_37}
    var_40 = module_0.Schema(var_39)
    var_41 = 'Nested'
    var_42 = module_0.Reference(var_41, var_33)
    var_43 = 123
    var_44 = {var_34: var_43}
    var_45 = {var_38: var_44}
    var_46 = var_42.validate(var_45)
    var_47 = module_0.Reference(var_5, var_0)
    var_48 = None
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'test'
    var_15 = {var_12: var_14}
    var_16 = var_11.validate(var_15)
    var_17 = module_1.Field(allow_null=var_10)
    var_18 = {var_12: var_17}
    var_19 = module_0.Schema(var_18)
    var_20 = 'StrictSchema'
    var_21 = module_0.Reference(var_20, var_0)
    var_22 = 'name'
    var_23 = None
    var_24 = {var_22: var_23}
    var_25 = var_21.validate(var_24)
    var_26 = {}
    var_27 = 'EmptySchema'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = 'extra'
    var_30 = 'field'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'invalid_key'
    var_21 = module_1.Field(allow_null=var_6)
    var_22 = 'name'
    var_23 = {var_22: var_21}
    var_24 = module_0.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'required'
    var_28 = module_1.Field()
    var_29 = {var_22: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'John'
    var_32 = {var_22: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = module_1.Field(read_only=var_26)
    var_35 = 'id'
    var_36 = {var_35: var_34}
    var_37 = module_0.Schema(var_36)
    var_38 = '123'
    var_39 = {var_35: var_38}
    var_40 = var_37.validate(var_39)
    var_41 = 'default_value'
    var_42 = module_1.Field(default=var_41)
    var_43 = 'status'
    var_44 = {var_43: var_42}
    var_45 = module_0.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)
    var_48 = module_1.Field()
    var_49 = 'Invalid'
    var_50 = 'invalid'
    var_51 = []
    var_52 = module_2.Message(text=var_49, code=var_50, index=var_51)
    var_53 = [var_52]
    var_54 = module_2.ValidationError(messages=var_53)
    var_55 = (var_18, var_54)
    var_56 = 'nested'
    var_57 = {var_56: var_48}
    var_58 = module_0.Schema(var_57)
    var_59 = 'nested'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = module_1.Field()
    var_64 = module_1.Field()
    var_65 = 'field1'
    var_66 = 'field2'
    var_67 = {var_65: var_63, var_66: var_64}
    var_68 = module_0.Schema(var_67)
    var_69 = 'value1'
    var_70 = 'value2'
    var_71 = {var_65: var_69, var_66: var_70}
    var_72 = var_68.validate(var_71)
    var_73 = 'key'
    var_74 = 'value'
    var_75 = {var_73: var_74}
    var_76 = module_1.Field()
    var_77 = {var_73: var_76}
    var_78 = module_0.Schema(var_77)
    var_79 = module_1.Field()
    var_80 = {var_22: var_79}
    var_81 = module_0.Schema(var_80)
    var_82 = 'extra'
    var_83 = 'field'
    var_84 = {var_22: var_31, var_82: var_83}
    var_85 = var_81.validate(var_84)
    var_86 = module_1.Field(allow_null=var_6)
    var_87 = module_1.Field(allow_null=var_6)
    var_88 = {var_65: var_86, var_66: var_87}
    var_89 = module_0.Schema(var_88)
    var_90 = {}
    var_91 = var_89.validate(var_90)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Reference(var_5, var_0)
    var_15 = 'test'
    var_16 = {var_12: var_15}
    var_17 = var_14.validate(var_16)
    var_18 = module_0.Reference(var_5, var_0)
    var_19 = 'invalid'
    var_20 = var_18.validate(var_19)
    var_21 = 'id'
    var_22 = 'value'
    var_23 = module_1.Field()
    var_24 = module_1.Field()
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'NestedSchema'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = '1'
    var_30 = 'data'
    var_31 = {var_21: var_29, var_22: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Reference(var_5, var_0)



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'John'
    var_15 = {var_12: var_14}
    var_16 = var_7.validate(var_15)
    var_17 = 10
    var_18 = module_1.String(max_length=var_17)
    var_19 = {var_12: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'SchemaWithRules'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 'Alice'
    var_24 = {var_12: var_23}
    var_25 = var_22.validate(var_24)
    var_26 = 'name'
    var_27 = 'ThisNameIsTooLongForTheRule'
    var_28 = {var_26: var_27}
    var_29 = var_22.validate(var_28)
    var_30 = {}
    var_31 = var_7.validate(var_30)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'John'
    var_14 = {var_11: var_13}
    var_15 = var_10.validate(var_14)
    var_16 = module_1.Field()
    var_17 = {var_11: var_16}
    var_18 = module_0.Schema(var_17)
    var_19 = 'SchemaWithField'
    var_20 = module_0.Reference(var_19, var_0)
    var_21 = 'Test'
    var_22 = {var_11: var_21}
    var_23 = var_20.validate(var_22)
    var_24 = module_0.Definitions()
    var_25 = 'id'
    var_26 = module_1.Field()
    var_27 = {var_25: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'TargetSchema'
    var_30 = module_0.Reference(var_29, var_24)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Definitions()
    var_15 = module_1.Field()
    var_16 = {var_12: var_15}
    var_17 = module_0.Schema(var_16)
    var_18 = 'Person'
    var_19 = module_0.Reference(var_18, var_14)
    var_20 = 'John'
    var_21 = {var_12: var_20}
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Definitions()
    var_24 = module_1.Field()
    var_25 = 'age'
    var_26 = {var_25: var_24}
    var_27 = module_0.Schema(var_26)
    var_28 = 'AgeSchema'
    var_29 = module_0.Reference(var_28, var_23)
    var_30 = 25
    var_31 = {var_25: var_30}
    var_32 = var_29.validate(var_31)
    var_33 = module_0.Definitions()
    var_34 = module_1.Field()
    var_35 = 'required_field'
    var_36 = {var_35: var_34}
    var_37 = module_0.Schema(var_36)
    var_38 = 'RequiredSchema'
    var_39 = module_0.Reference(var_38, var_33)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Reference(var_5, var_0)
    var_43 = None
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = {var_1: var_0}
    var_8 = False
    var_9 = module_1.Schema(var_7)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = {var_10: var_0}
    var_13 = module_1.Schema(var_12)
    var_14 = 'invalid'
    var_15 = var_13.validate(var_14)
    var_16 = {var_14: var_0}
    var_17 = module_1.Schema(var_16)
    var_18 = 1
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = 'invalid_key'
    var_23 = module_0.Field()
    var_24 = {var_18: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = 'required'
    var_29 = module_0.Field(read_only=var_20)
    var_30 = 'id'
    var_31 = {var_30: var_29}
    var_32 = module_1.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = 'default_value'
    var_36 = module_0.Field(default=var_35)
    var_37 = 'status'
    var_38 = {var_37: var_36}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field()
    var_43 = {var_26: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = 'John'
    var_46 = {var_26: var_45}
    var_47 = var_44.validate(var_46)
    var_48 = module_0.Field()
    var_49 = 'Invalid'
    var_50 = 'invalid'
    var_51 = []
    var_52 = module_2.Message(text=var_49, code=var_50, index=var_51)
    var_53 = [var_52]
    var_54 = module_2.ValidationError(messages=var_53)
    var_55 = (var_21, var_54)
    var_56 = 'data'
    var_57 = {var_56: var_48}
    var_58 = module_1.Schema(var_57)
    var_59 = 'data'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = module_0.Field()
    var_64 = 'key'
    var_65 = {var_64: var_63}
    var_66 = module_1.Schema(var_65)
    var_67 = 'value'
    var_68 = (var_64, var_67)
    var_69 = [var_68]
    var_70 = module_0.Field()
    var_71 = module_0.Field()
    var_72 = 'a'
    var_73 = 'b'
    var_74 = {var_72: var_70, var_73: var_71}
    var_75 = module_1.Schema(var_74)
    var_76 = {}
    var_77 = var_75.validate(var_76)
    var_78 = module_0.Field()
    var_79 = module_0.Field()
    var_80 = 'Error'
    var_81 = 'error'
    var_82 = []
    var_83 = module_2.Message(text=var_80, code=var_81, index=var_82)
    var_84 = [var_83]
    var_85 = module_2.ValidationError(messages=var_84)
    var_86 = (var_62, var_85)
    var_87 = 'valid'
    var_88 = {var_87: var_78, var_50: var_79}
    var_89 = module_1.Schema(var_88)
    var_90 = 'valid'
    var_91 = 'invalid'
    var_92 = 'data'
    var_93 = {var_90: var_92, var_91: var_92}
    var_94 = var_89.validate(var_93)
    var_95 = module_0.Field(read_only=var_92)
    var_96 = 'default'
    var_97 = module_0.Field(default=var_96)
    var_98 = 'readonly'
    var_99 = 'withdefault'
    var_100 = {var_98: var_95, var_99: var_97}
    var_101 = module_1.Schema(var_100)
    var_102 = {}
    var_103 = var_101.validate(var_102)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = 'test'
    var_15 = {var_11: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = 5
    var_18 = module_1.String(max_length=var_17)
    var_19 = {var_11: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'StringSchema'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 'name'
    var_24 = 'this is too long'
    var_25 = {var_23: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = 'id'
    var_28 = module_1.Field()
    var_29 = {var_27: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'SimpleSchema'
    var_32 = module_0.Reference(var_31, var_0)
    var_33 = 123
    var_34 = {var_27: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_0.Reference(var_26, var_0)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'name'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = {var_21: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'John'
    var_30 = {var_21: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_1.Field(read_only=var_25)
    var_33 = 'id'
    var_34 = {var_33: var_32}
    var_35 = module_0.Schema(var_34)
    var_36 = {}
    var_37 = var_35.validate(var_36)
    var_38 = 'default_value'
    var_39 = module_1.Field(default=var_38)
    var_40 = 'status'
    var_41 = {var_40: var_39}
    var_42 = module_0.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = 5
    var_46 = module_1.String(max_length=var_45)
    var_47 = {var_21: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = 'name'
    var_50 = 'toolongname'
    var_51 = {var_49: var_50}
    var_52 = var_48.validate(var_51)
    var_53 = module_1.Field()
    var_54 = module_1.Field()
    var_55 = 'field1'
    var_56 = 'field2'
    var_57 = {var_55: var_53, var_56: var_54}
    var_58 = module_0.Schema(var_57)
    var_59 = 'field1'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = module_1.Field()
    var_64 = {var_21: var_63}
    var_65 = module_0.Schema(var_64)
    var_66 = (var_21, var_29)
    var_67 = [var_66]
    var_68 = module_1.Field()
    var_69 = {var_21: var_68}
    var_70 = module_0.Schema(var_69)
    var_71 = 'extra'
    var_72 = 'ignored'
    var_73 = {var_21: var_29, var_71: var_72}
    var_74 = var_70.validate(var_73)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_16: var_18, var_17: var_18}
    var_20 = var_15.validate(var_19)
    var_21 = 'invalid_key'
    var_22 = module_1.Field()
    var_23 = 'required_field'
    var_24 = {var_23: var_22}
    var_25 = module_0.Schema(var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = 'required'
    var_29 = module_1.Field()
    var_30 = 'key'
    var_31 = {var_30: var_29}
    var_32 = module_0.Schema(var_31)
    var_33 = 'value'
    var_34 = {var_30: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_1.Field(read_only=var_27)
    var_37 = 'read_only'
    var_38 = {var_37: var_36}
    var_39 = module_0.Schema(var_38)
    var_40 = {var_37: var_33}
    var_41 = var_39.validate(var_40)
    var_42 = 'default_value'
    var_43 = module_1.Field(default=var_42)
    var_44 = 'with_default'
    var_45 = {var_44: var_43}
    var_46 = module_0.Schema(var_45)
    var_47 = {}
    var_48 = var_46.validate(var_47)
    var_49 = module_1.Field()
    var_50 = ()
    var_51 = 'child_error'
    var_52 = module_2.ValidationError(code=var_51)
    var_53 = 'child'
    var_54 = {var_53: var_49}
    var_55 = module_0.Schema(var_54)
    var_56 = 'child'
    var_57 = 'value'
    var_58 = {var_56: var_57}
    var_59 = var_55.validate(var_58)
    var_60 = module_1.Field()
    var_61 = {var_30: var_60}
    var_62 = module_0.Schema(var_61)
    var_63 = {var_30: var_33}
    var_64 = var_62.validate(var_63)
    var_65 = module_1.Field()
    var_66 = module_1.Field()
    var_67 = 'field1'
    var_68 = 'field2'
    var_69 = {var_67: var_65, var_68: var_66}
    var_70 = module_0.Schema(var_69)
    var_71 = 1
    var_72 = 'value'
    var_73 = {var_71: var_72}
    var_74 = var_70.validate(var_73)
    var_75 = {}
    var_76 = module_0.Schema(var_75)
    var_77 = {}
    var_78 = var_76.validate(var_77)



# Parsed testcases at query #22
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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = 0
    var_22 = exc_info.value.messages()[var_21]
    var_23 = var_22.code
    assert var_23 == 'null'
    var_24 = module_0.String()
    var_25 = {var_19: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'not a dict'
    var_28 = var_26.validate(var_27)
    var_29 = exc_info.value.messages()[var_21]
    var_30 = var_29.code
    assert var_30 == 'type'
    var_31 = module_0.String()
    var_32 = {var_27: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 1
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = var_33.validate(var_36)
    var_38 = exc_info.value.messages()[var_21]
    var_39 = var_38.code
    assert var_39 == 'invalid_key'
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = {var_34: var_40, var_35: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = 'name'
    var_45 = 'John'
    var_46 = {var_44: var_45}
    var_47 = var_43.validate(var_46)
    var_48 = 'required'
    var_49 = 'status'
    var_50 = module_0.String()
    var_51 = 'active'
    var_52 = module_0.String()
    var_53 = {var_44: var_50, var_49: var_52}
    var_54 = module_1.Schema(var_53)
    var_55 = {var_44: var_6}
    var_56 = var_54.validate(var_55)
    var_57 = 'id'
    var_58 = module_0.String()
    var_59 = module_0.Integer()
    var_60 = {var_44: var_58, var_57: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = 123
    var_63 = {var_44: var_6, var_57: var_62}
    var_64 = var_61.validate(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Integer()
    var_67 = {var_44: var_65, var_45: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = 'name'
    var_70 = 'age'
    var_71 = 'John'
    var_72 = 'not an integer'
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = var_68.validate(var_73)
    var_75 = module_0.String()
    var_76 = {var_69: var_75}
    var_77 = module_1.Schema(var_76)
    var_78 = (var_69, var_74)
    var_79 = [var_78]
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = {var_69: var_80, var_70: var_81}
    var_83 = module_1.Schema(var_82)
    var_84 = 1
    var_85 = 'age'
    var_86 = 'value'
    var_87 = 'invalid'
    var_88 = {var_84: var_86, var_85: var_87}
    var_89 = var_83.validate(var_88)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = True
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Reference(var_5, var_0)
    var_17 = 'another_test'
    var_18 = {var_10: var_17}
    var_19 = var_16.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'StringField'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 'test_string'
    var_24 = var_22.validate(var_23)
    assert var_24 == 'test_string'



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Field()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Field()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Field()
    var_22 = {var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'invalid'
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
    var_44 = module_0.Field(read_only=var_12)
    var_45 = 'id'
    var_46 = {var_45: var_44}
    var_47 = module_1.Schema(var_46)
    var_48 = 123
    var_49 = {var_45: var_48}
    var_50 = var_47.validate(var_49)
    var_51 = module_0.Integer()
    var_52 = {var_37: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = 'age'
    var_55 = 'not_an_int'
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = module_0.Field()
    var_59 = {var_54: var_58}
    var_60 = module_1.Schema(var_59)
    var_61 = (var_54, var_6)
    var_62 = [var_61]
    var_63 = {}
    var_64 = module_1.Schema(var_63)
    var_65 = {}
    var_66 = var_64.validate(var_65)
    var_67 = module_0.Field()
    var_68 = {var_54: var_67}
    var_69 = module_1.Schema(var_68)
    var_70 = 'extra'
    var_71 = 'field'
    var_72 = {var_54: var_6, var_70: var_71}
    var_73 = var_69.validate(var_72)
    var_74 = module_0.Field()
    var_75 = module_0.Field()
    var_76 = {var_54: var_74, var_55: var_75}
    var_77 = module_1.Schema(var_76)
    var_78 = {}
    var_79 = var_77.validate(var_78)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Field()
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_1.Schema(var_8)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_9]
    var_14 = var_13.code
    assert var_14 == 'null'
    var_15 = module_0.Field()
    var_16 = {var_11: var_15}
    var_17 = module_1.Schema(var_16)
    var_18 = 'not a dict'
    var_19 = var_17.validate(var_18)
    var_20 = exc_info.value.messages()[var_9]
    var_21 = var_20.code
    assert var_21 == 'type'
    var_22 = module_0.Field()
    var_23 = {var_18: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'John'
    var_26 = {var_18: var_25}
    var_27 = var_24.validate(var_26)
    var_28 = module_0.Field()
    var_29 = {var_18: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 1
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'invalid_key'
    var_36 = module_0.Field(allow_null=var_9)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = 'required'
    var_42 = 'Unknown'
    var_43 = module_0.Field(default=var_42)
    var_44 = {var_39: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Field(read_only=var_34)
    var_49 = {var_39: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = {var_39: var_25}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Field()
    var_54 = 'value'
    var_55 = {var_54: var_53}
    var_56 = module_1.Schema(var_55)
    var_57 = 123
    var_58 = {var_54: var_57}
    var_59 = var_56.validate(var_58)
    var_60 = 'Alice'
    var_61 = (var_39, var_60)
    var_62 = [var_61]
    var_63 = module_0.Field()
    var_64 = {var_39: var_63}
    var_65 = module_1.Schema(var_64)
    var_66 = module_0.Field()
    var_67 = module_0.Field()
    var_68 = 'field1'
    var_69 = 'field2'
    var_70 = {var_68: var_66, var_69: var_67}
    var_71 = module_1.Schema(var_70)
    var_72 = {}
    var_73 = var_71.validate(var_72)
    var_74 = module_0.Field()
    var_75 = {var_72: var_74}
    var_76 = module_1.Schema(var_75)
    var_77 = 'extra'
    var_78 = 'field'
    var_79 = {var_72: var_25, var_77: var_78}
    var_80 = var_76.validate(var_79)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = None
    var_19 = var_6.validate(var_18)
    var_20 = module_0.Definitions()
    var_21 = 'id'
    var_22 = module_1.Field()
    var_23 = {var_21: var_22}
    var_24 = module_0.Schema(var_23)
    var_25 = module_1.Field()
    var_26 = {var_18: var_25}
    var_27 = module_0.Schema(var_26)
    var_28 = 'Schema1'
    var_29 = module_0.Reference(var_28, var_20)
    var_30 = 'Schema2'
    var_31 = module_0.Reference(var_30, var_20)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Reference(var_5, var_0)
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = 0
    var_18 = exc_info.value.messages()[var_17]
    var_19 = var_18.code
    assert var_19 == 'null'
    var_20 = module_0.Reference(var_5, var_0)
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = 'id'
    var_24 = 'email'
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_23: var_25, var_22: var_26, var_24: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = 'ComplexSchema'
    var_31 = module_0.Reference(var_30, var_0)
    var_32 = 'John'
    var_33 = 'john@example.com'
    var_34 = {var_23: var_10, var_22: var_32, var_24: var_33}
    var_35 = var_31.validate(var_34)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = True
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Reference(var_5, var_0)
    var_17 = 'John'
    var_18 = {var_10: var_17}
    var_19 = var_16.validate(var_18)
    var_20 = 'age'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = 'InvalidSchema'
    var_25 = module_0.Reference(var_24, var_0)
    var_26 = 'not a dict'
    var_27 = var_25.validate(var_26)



# Parsed testcases at query #29
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'required_key'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = 'key'
    var_28 = {var_27: var_26}
    var_29 = module_0.Schema(var_28)
    var_30 = 'value'
    var_31 = {var_27: var_30}
    var_32 = var_29.validate(var_31)
    var_33 = 'default_value'
    var_34 = module_1.Field(default=var_33)
    var_35 = {var_27: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = module_1.Field(read_only=var_25)
    var_40 = 'readonly'
    var_41 = {var_40: var_39}
    var_42 = module_0.Schema(var_41)
    var_43 = 'should_be_ignored'
    var_44 = {var_40: var_43}
    var_45 = var_42.validate(var_44)
    var_46 = module_1.Field()
    var_47 = 'default'
    var_48 = module_1.Field(default=var_47)
    var_49 = 'field1'
    var_50 = 'field2'
    var_51 = {var_49: var_46, var_50: var_48}
    var_52 = module_0.Schema(var_51)
    var_53 = 'value1'
    var_54 = {var_49: var_53}
    var_55 = var_52.validate(var_54)
    var_56 = 'nested'
    var_57 = module_1.Field()
    var_58 = {var_56: var_57}
    var_59 = module_0.Schema(var_58)
    var_60 = 'child'
    var_61 = {var_60: var_59}
    var_62 = module_0.Schema(var_61)
    var_63 = 'child'
    var_64 = 'invalid'
    var_65 = {var_63: var_64}
    var_66 = var_62.validate(var_65)
    var_67 = {}
    var_68 = module_0.Schema(var_67)
    var_69 = {}
    var_70 = var_68.validate(var_69)
    var_71 = module_1.Field()
    var_72 = {var_27: var_71}
    var_73 = module_0.Schema(var_72)
    var_74 = (var_27, var_30)
    var_75 = [var_74]



# Parsed testcases at query #30
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = 'TestSchema'
    var_4 = True
    var_5 = module_0.Reference(var_3, var_0)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = False
    var_9 = module_0.Reference(var_3, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_1.Field()
    var_13 = 'SimpleField'
    var_14 = module_0.Reference(var_13, var_0)
    var_15 = 'test_value'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'test_value'
    var_17 = 'name'
    var_18 = 'age'
    var_19 = module_1.Field()
    var_20 = module_1.Field()
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.Schema(var_21)
    var_23 = 'ComplexSchema'
    var_24 = module_0.Reference(var_23, var_0)
    var_25 = 'John'
    var_26 = 30
    var_27 = {var_17: var_25, var_18: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = 'required_field'
    var_30 = module_1.Field()
    var_31 = {var_29: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'StrictSchema'
    var_34 = module_0.Reference(var_33, var_0)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 'value'
    var_38 = module_1.Field()
    var_39 = {var_37: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = 'InnerSchema'
    var_42 = module_0.Reference(var_41, var_0)
    var_43 = 'test'
    var_44 = {var_37: var_43}
    var_45 = var_42.validate(var_44)



# Parsed testcases at query #31
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'invalid_key'
    var_21 = 'name'
    var_22 = module_1.Field(allow_null=var_6)
    var_23 = {var_21: var_22}
    var_24 = module_0.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'required'
    var_28 = module_1.Field()
    var_29 = {var_21: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'John'
    var_32 = {var_21: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = module_1.Field()
    var_35 = {var_21: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = module_1.Field(read_only=var_26)
    var_40 = 'id'
    var_41 = {var_40: var_39}
    var_42 = module_0.Schema(var_41)
    var_43 = '123'
    var_44 = {var_40: var_43}
    var_45 = var_42.validate(var_44)
    var_46 = module_1.Field()
    var_47 = 'age'
    var_48 = {var_47: var_46}
    var_49 = module_0.Schema(var_48)
    var_50 = 'age'
    var_51 = 'invalid'
    var_52 = {var_50: var_51}
    var_53 = var_49.validate(var_52)
    var_54 = 'email'
    var_55 = module_1.Field()
    var_56 = module_1.Field()
    var_57 = {var_21: var_55, var_54: var_56}
    var_58 = module_0.Schema(var_57)
    var_59 = 'john@example.com'
    var_60 = {var_21: var_31, var_54: var_59}
    var_61 = var_58.validate(var_60)
    var_62 = module_1.Field()
    var_63 = {var_21: var_62}
    var_64 = module_0.Schema(var_63)
    var_65 = (var_21, var_31)
    var_66 = [var_65]
    var_67 = 'required_field'
    var_68 = 'another_required'
    var_69 = module_1.Field()
    var_70 = module_1.Field()
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = module_0.Schema(var_71)
    var_73 = {}
    var_74 = var_72.validate(var_73)
    var_75 = module_1.Field()
    var_76 = {var_21: var_75}
    var_77 = module_0.Schema(var_76)
    var_78 = 'extra'
    var_79 = 'field'
    var_80 = {var_21: var_31, var_78: var_79}
    var_81 = var_77.validate(var_80)



# Parsed testcases at query #32
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = module_1.Field()
    var_15 = {var_11: var_14}
    var_16 = module_0.Schema(var_15)
    var_17 = module_0.Reference(var_5, var_13)
    var_18 = 'test'
    var_19 = {var_11: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = module_0.Definitions()
    var_22 = 5
    var_23 = module_1.String(max_length=var_22)
    var_24 = {var_11: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = module_0.Reference(var_5, var_21)
    var_27 = 'name'
    var_28 = 'this is a very long string'
    var_29 = {var_27: var_28}
    var_30 = var_26.validate(var_29)
    var_31 = module_0.Definitions()
    var_32 = module_1.Field()
    var_33 = {var_27: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = module_0.Reference(var_30, var_31)
    var_36 = {}
    var_37 = var_35.validate(var_36)
    var_38 = module_0.Definitions()
    var_39 = module_1.Field()
    var_40 = {var_36: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = module_0.Reference(var_30, var_38)
    var_43 = 'not a dict'
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #33
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = None
    var_19 = var_6.validate(var_18)
    var_20 = 'id'
    var_21 = 'data'
    var_22 = module_1.Field()
    var_23 = module_1.Field()
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = 'ComplexSchema'
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = 'value'
    var_29 = {var_20: var_10, var_21: var_28}
    var_30 = var_27.validate(var_29)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Field()
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_1.Schema(var_8)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Field()
    var_14 = {var_11: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = 'not a dict'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Field()
    var_19 = {var_16: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 1
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = 'invalid_key'
    var_26 = module_0.Field()
    var_27 = {var_21: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = 'required'
    var_32 = module_0.Field()
    var_33 = {var_29: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = 'John'
    var_36 = {var_29: var_35}
    var_37 = var_34.validate(var_36)
    var_38 = 'default_value'
    var_39 = module_0.Field(default=var_38)
    var_40 = {var_29: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Field(read_only=var_24)
    var_45 = 'id'
    var_46 = module_0.Field()
    var_47 = {var_45: var_44, var_29: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_45: var_24, var_29: var_35}
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Field()
    var_52 = 'Invalid'
    var_53 = 'invalid'
    var_54 = []
    var_55 = module_2.Message(text=var_52, code=var_53, index=var_54)
    var_56 = [var_55]
    var_57 = module_2.ValidationError(messages=var_56)
    var_58 = (var_5, var_57)
    var_59 = 'nested'
    var_60 = {var_59: var_51}
    var_61 = module_1.Schema(var_60)
    var_62 = 'nested'
    var_63 = 'invalid'
    var_64 = {var_62: var_63}
    var_65 = var_61.validate(var_64)
    var_66 = module_0.Field()
    var_67 = {var_62: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = (var_62, var_35)
    var_70 = [var_69]
    var_71 = module_0.Field()
    var_72 = module_0.Field(allow_null=var_65)
    var_73 = 'optional'
    var_74 = {var_31: var_71, var_73: var_72}
    var_75 = module_1.Schema(var_74)
    var_76 = 'value'
    var_77 = {var_31: var_76}
    var_78 = var_75.validate(var_77)
    var_79 = 'default'
    var_80 = module_0.Field(default=var_79, read_only=var_65)
    var_81 = {var_45: var_80}
    var_82 = module_1.Schema(var_81)
    var_83 = {}
    var_84 = var_82.validate(var_83)



# Parsed testcases at query #35
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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = {var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.String()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 1
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'invalid_key'
    var_34 = module_0.String()
    var_35 = {var_29: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'required'
    var_40 = [var_37]
    var_41 = module_0.String()
    var_42 = 0
    var_43 = module_0.Integer()
    var_44 = {var_37: var_41, var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {var_37: var_6}
    var_47 = var_45.validate(var_46)
    var_48 = module_0.String()
    var_49 = {var_37: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.String()
    var_54 = module_0.Integer()
    var_55 = {var_37: var_53, var_38: var_54}
    var_56 = module_1.Schema(var_55)
    var_57 = 'name'
    var_58 = 'age'
    var_59 = 'John'
    var_60 = 'not_an_int'
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = var_56.validate(var_61)
    var_63 = [var_58]
    var_64 = module_0.String()
    var_65 = module_0.Integer()
    var_66 = {var_57: var_64, var_58: var_65}
    var_67 = module_1.Schema(var_66)
    var_68 = 1
    var_69 = 'age'
    var_70 = 'value'
    var_71 = 'invalid'
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = var_67.validate(var_72)
    var_74 = module_0.String()
    var_75 = {var_68: var_74}
    var_76 = module_1.Schema(var_75)
    var_77 = {var_68: var_73}
    var_78 = module_0.String()
    var_79 = {var_68: var_78}
    var_80 = module_1.Schema(var_79)
    var_81 = 'extra'
    var_82 = 'field'
    var_83 = {var_68: var_73, var_81: var_82}
    var_84 = var_80.validate(var_83)



# Parsed testcases at query #36
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'required_field'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = 'field'
    var_28 = {var_27: var_26}
    var_29 = module_0.Schema(var_28)
    var_30 = 'value'
    var_31 = {var_27: var_30}
    var_32 = var_29.validate(var_31)
    var_33 = 'default_value'
    var_34 = module_1.Field(default=var_33)
    var_35 = {var_27: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = module_1.Field(read_only=var_25)
    var_40 = 'readonly'
    var_41 = {var_40: var_39}
    var_42 = module_0.Schema(var_41)
    var_43 = 'should_be_ignored'
    var_44 = {var_40: var_43}
    var_45 = var_42.validate(var_44)
    var_46 = module_1.Field()
    var_47 = ()
    var_48 = 'Inner error'
    var_49 = 'inner'
    var_50 = []
    var_51 = module_2.Message(text=var_48, code=var_49, index=var_50)
    var_52 = [var_51]
    var_53 = module_2.ValidationError(messages=var_52)
    var_54 = 'nested'
    var_55 = {var_54: var_46}
    var_56 = module_0.Schema(var_55)
    var_57 = 'nested'
    var_58 = 'value'
    var_59 = {var_57: var_58}
    var_60 = var_56.validate(var_59)
    var_61 = {}
    var_62 = module_0.Schema(var_61)
    var_63 = {}
    var_64 = var_62.validate(var_63)
    var_65 = module_1.Field()
    var_66 = {var_27: var_65}
    var_67 = module_0.Schema(var_66)
    var_68 = (var_27, var_30)
    var_69 = [var_68]
    var_70 = module_1.Field()
    var_71 = module_1.Field()
    var_72 = 'field1'
    var_73 = 'field2'
    var_74 = {var_72: var_70, var_73: var_71}
    var_75 = module_0.Schema(var_74)
    var_76 = 'field1'
    var_77 = 'value'
    var_78 = {var_76: var_77}
    var_79 = var_75.validate(var_78)
    var_80 = [msg for msg in e.messages() if msg.code == 'required']



# Parsed testcases at query #37
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'required_field'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = 'key'
    var_28 = {var_27: var_26}
    var_29 = module_0.Schema(var_28)
    var_30 = 'value'
    var_31 = {var_27: var_30}
    var_32 = var_29.validate(var_31)
    var_33 = 'default_value'
    var_34 = module_1.Field(default=var_33)
    var_35 = 'field'
    var_36 = {var_35: var_34}
    var_37 = module_0.Schema(var_36)
    var_38 = {}
    var_39 = var_37.validate(var_38)
    var_40 = module_1.Field(read_only=var_25)
    var_41 = 'read_only'
    var_42 = {var_41: var_40}
    var_43 = module_0.Schema(var_42)
    var_44 = {var_41: var_30}
    var_45 = var_43.validate(var_44)
    var_46 = module_1.Field()
    var_47 = {var_27: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = (var_27, var_30)
    var_50 = [var_49]
    var_51 = 'invalid'
    var_52 = 'Invalid value'
    var_53 = {var_51: var_52}
    var_54 = 'strict'
    var_55 = 'strict'
    var_56 = 'invalid'
    var_57 = {var_55: var_56}
    var_58 = var_48.validate(var_57)
    var_59 = module_1.Field()
    var_60 = 'field1'
    var_61 = 'field1'
    var_62 = 'strict'
    var_63 = 'value1'
    var_64 = 'invalid'
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = var_48.validate(var_65)
    var_67 = {}
    var_68 = module_0.Schema(var_67)
    var_69 = {}
    var_70 = var_68.validate(var_69)
    var_71 = module_1.Field()
    var_72 = {var_27: var_71}
    var_73 = module_0.Schema(var_72)
    var_74 = 'extra'
    var_75 = 'extra_value'
    var_76 = {var_27: var_30, var_74: var_75}
    var_77 = var_73.validate(var_76)



# Parsed testcases at query #38
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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = {var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.String()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 1
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'invalid_key'
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_29: var_34, var_30: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'required'
    var_43 = 'status'
    var_44 = module_0.String()
    var_45 = 'active'
    var_46 = module_0.String()
    var_47 = {var_38: var_44, var_43: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_38: var_6}
    var_50 = var_48.validate(var_49)
    var_51 = 'id'
    var_52 = module_0.String()
    var_53 = module_0.String()
    var_54 = {var_38: var_52, var_51: var_53}
    var_55 = module_1.Schema(var_54)
    var_56 = '123'
    var_57 = {var_38: var_6, var_51: var_56}
    var_58 = var_55.validate(var_57)
    var_59 = module_0.String()
    var_60 = module_0.Integer()
    var_61 = {var_38: var_59, var_39: var_60}
    var_62 = module_1.Schema(var_61)
    var_63 = 'name'
    var_64 = 'age'
    var_65 = 'John'
    var_66 = 'not_an_int'
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = var_62.validate(var_67)
    var_69 = 'email'
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = module_0.String()
    var_73 = {var_63: var_70, var_64: var_71, var_69: var_72}
    var_74 = module_1.Schema(var_73)
    var_75 = 1
    var_76 = 'age'
    var_77 = 'value'
    var_78 = 'invalid'
    var_79 = {var_75: var_77, var_76: var_78}
    var_80 = var_74.validate(var_79)
    var_81 = 'optional_field'
    var_82 = 'default'
    var_83 = module_0.String()
    var_84 = {var_81: var_83}
    var_85 = module_1.Schema(var_84)
    var_86 = {}
    var_87 = var_85.validate(var_86)
    var_88 = module_0.String()
    var_89 = {var_75: var_88}
    var_90 = module_1.Schema(var_89)
    var_91 = (var_75, var_80)
    var_92 = [var_91]



# Parsed testcases at query #39
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = 'test'
    var_15 = {var_11: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = module_0.Reference(var_5, var_0)
    var_18 = 'invalid'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Reference(var_5, var_0)
    var_21 = 'id'
    var_22 = module_1.Field()
    var_23 = module_1.Field()
    var_24 = {var_21: var_22, var_18: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = 'ComplexSchema'
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = {var_21: var_6, var_18: var_14}
    var_29 = var_27.validate(var_28)



# Parsed testcases at query #40
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Definitions()
    var_15 = module_1.Field()
    var_16 = {var_12: var_15}
    var_17 = module_0.Schema(var_16)
    var_18 = 'Person'
    var_19 = module_0.Reference(var_18, var_14)
    var_20 = 'John'
    var_21 = {var_12: var_20}
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Definitions()
    var_24 = module_1.Field(allow_null=var_10)
    var_25 = 'required_field'
    var_26 = {var_25: var_24}
    var_27 = module_0.Schema(var_26)
    var_28 = 'StrictSchema'
    var_29 = module_0.Reference(var_28, var_23)
    var_30 = {var_25: var_8}
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = {}
    var_34 = module_0.Schema(var_33)
    var_35 = 'EmptySchema'
    var_36 = module_0.Reference(var_35, var_32)
    var_37 = {}
    var_38 = var_36.validate(var_37)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = {var_0: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.String()
    var_13 = {var_10: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = 'invalid'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = {var_15: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = 'invalid_key'
    var_25 = module_0.String()
    var_26 = {var_20: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = 'required'
    var_31 = 'age'
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = {var_28: var_32, var_31: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = 'John'
    var_37 = 30
    var_38 = {var_28: var_36, var_31: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = module_0.String()
    var_41 = 'email'
    var_42 = module_0.String()
    var_43 = {var_28: var_42, var_41: var_40}
    var_44 = module_1.Schema(var_43)
    var_45 = {var_28: var_36}
    var_46 = var_44.validate(var_45)
    var_47 = 'active'
    var_48 = module_0.String()
    var_49 = module_0.Boolean()
    var_50 = {var_28: var_48, var_47: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_28: var_36}
    var_53 = var_51.validate(var_52)
    var_54 = 'id'
    var_55 = module_0.String()
    var_56 = module_0.String()
    var_57 = {var_28: var_55, var_54: var_56}
    var_58 = module_1.Schema(var_57)
    var_59 = '123'
    var_60 = {var_28: var_36, var_54: var_59}
    var_61 = var_58.validate(var_60)
    var_62 = module_0.String()
    var_63 = module_0.Integer()
    var_64 = {var_28: var_62, var_31: var_63}
    var_65 = module_1.Schema(var_64)
    var_66 = 'name'
    var_67 = 'age'
    var_68 = 'John'
    var_69 = 'not_an_int'
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = var_65.validate(var_70)
    var_72 = 'Unknown'
    var_73 = module_0.String()
    var_74 = 0
    var_75 = module_0.Integer()
    var_76 = {var_66: var_73, var_31: var_75}
    var_77 = module_1.Schema(var_76)
    var_78 = {}
    var_79 = var_77.validate(var_78)
    var_80 = module_0.String()
    var_81 = {var_66: var_80}
    var_82 = module_1.Schema(var_81)
    var_83 = 'extra'
    var_84 = 'field'
    var_85 = {var_66: var_36, var_83: var_84}
    var_86 = var_82.validate(var_85)
    var_87 = module_0.String()
    var_88 = {var_66: var_87}
    var_89 = module_1.Schema(var_88)
    var_90 = (var_66, var_36)
    var_91 = [var_90]
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = {var_66: var_92, var_31: var_93}
    var_95 = module_1.Schema(var_94)
    var_96 = {}
    var_97 = var_95.validate(var_96)



# Parsed testcases at query #42
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = var_11.validate(var_17)
    var_19 = {}
    var_20 = module_0.Schema(var_19)
    var_21 = {}
    var_22 = var_20.validate(var_21)
    var_23 = {}
    var_24 = module_0.Schema(var_23)
    var_25 = 1
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = 'invalid_key'
    var_30 = module_1.Field()
    var_31 = 'name'
    var_32 = {var_31: var_30}
    var_33 = module_0.Schema(var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = 'required'
    var_37 = module_1.Field()
    var_38 = {var_31: var_37}
    var_39 = module_0.Schema(var_38)
    var_40 = 'John'
    var_41 = {var_31: var_40}
    var_42 = var_39.validate(var_41)
    var_43 = 'default_value'
    var_44 = module_1.Field(default=var_43)
    var_45 = 'status'
    var_46 = {var_45: var_44}
    var_47 = module_0.Schema(var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)
    var_50 = module_1.Field(read_only=var_35)
    var_51 = 'id'
    var_52 = {var_51: var_50}
    var_53 = module_0.Schema(var_52)
    var_54 = 123
    var_55 = {var_51: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = module_1.Field()
    var_58 = 'nested'
    var_59 = {var_58: var_57}
    var_60 = module_0.Schema(var_59)
    var_61 = 'nested'
    var_62 = None
    var_63 = {var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = 'age'
    var_66 = module_1.Field()
    var_67 = module_1.Field()
    var_68 = {var_31: var_66, var_65: var_67}
    var_69 = module_0.Schema(var_68)
    var_70 = 30
    var_71 = {var_31: var_40, var_65: var_70}
    var_72 = var_69.validate(var_71)
    var_73 = module_1.Field()
    var_74 = {var_31: var_73}
    var_75 = module_0.Schema(var_74)
    var_76 = 'extra'
    var_77 = 'field'
    var_78 = {var_31: var_40, var_76: var_77}
    var_79 = var_75.validate(var_78)
    var_80 = 'Jane'
    var_81 = (var_31, var_80)
    var_82 = [var_81]
    var_83 = module_1.Field()
    var_84 = {var_31: var_83}
    var_85 = module_0.Schema(var_84)
    var_86 = 'field1'
    var_87 = 'field2'
    var_88 = module_1.Field()
    var_89 = module_1.Field()
    var_90 = {var_86: var_88, var_87: var_89}
    var_91 = module_0.Schema(var_90)
    var_92 = 'field1'
    var_93 = None
    var_94 = {var_92: var_93}
    var_95 = var_91.validate(var_94)



# Parsed testcases at query #43
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = exc_info.value.messages()[var_14]
    var_19 = var_18.code
    assert var_19 == 'null'
    var_20 = module_0.Reference(var_5, var_0)
    var_21 = 'John'
    var_22 = {var_16: var_21}
    var_23 = var_20.validate(var_22)
    var_24 = 'not a dict'
    var_25 = var_20.validate(var_24)



# Parsed testcases at query #44
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = 'test'
    var_14 = {var_7: var_13}
    var_15 = var_6.validate(var_14)
    var_16 = module_0.Definitions()
    var_17 = False
    var_18 = module_1.Field(allow_null=var_17)
    var_19 = {var_7: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'RequiredSchema'
    var_22 = module_0.Reference(var_21, var_16)
    var_23 = 'John'
    var_24 = {var_7: var_23}
    var_25 = var_22.validate(var_24)
    var_26 = {}
    var_27 = var_6.validate(var_26)



# Parsed testcases at query #45
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'test'
    var_14 = {var_11: var_13}
    var_15 = var_7.validate(var_14)
    var_16 = None
    var_17 = var_10.validate(var_16)
    var_18 = module_0.Definitions()
    var_19 = module_1.Field()
    var_20 = 'SimpleField'
    var_21 = module_0.Reference(var_20, var_18)
    var_22 = 'test_value'
    var_23 = var_21.validate(var_22)
    assert var_23 == 'test_value'



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Test __setitem__ method of Definitions class.'
    var_1 = module_0.Definitions()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'initial'
    var_8 = {var_7: var_3}



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'nested'
    var_2 = 'dict'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    var_8 = 'id'
    var_9 = module_0.Field()
    var_10 = {var_8: var_7, var_0: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'default_value'
    var_13 = module_0.Field(default=var_12)
    var_14 = 'status'
    var_15 = module_0.Field()
    var_16 = {var_0: var_15, var_14: var_13}
    var_17 = module_1.Schema(var_16)
    var_18 = 'email'
    var_19 = module_0.Field(read_only=var_6)
    var_20 = module_0.Field()
    var_21 = 'active'
    var_22 = module_0.Field(default=var_21)
    var_23 = module_0.Field()
    var_24 = {var_8: var_19, var_0: var_20, var_14: var_22, var_18: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = var_25.required
    var_27 = set(var_26)
    var_28 = module_0.Field(read_only=var_6)
    var_29 = module_0.Field(default=var_21)
    var_30 = {var_8: var_28, var_14: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = module_1.Schema(var_4)
    var_33 = {}
    var_34 = module_1.Schema(var_33)
    var_35 = module_1.Schema(var_4)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Test __setitem__ method of Definitions class.'
    var_1 = module_0.Definitions()
    var_2 = len(var_1)
    assert var_2 == 2
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'nested'
    var_7 = 'dict'
    var_8 = 4
    var_9 = 5
    var_10 = 6
    var_11 = 'initial_key'
    var_12 = 'initial_value'
    var_13 = {var_11: var_12}



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'active'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = module_0.Boolean()
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = True
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_10}
    var_12 = var_7.serialize(var_11)
    var_13 = None
    var_14 = var_7.serialize(var_13)
    assert var_14 is None
    var_15 = 'Jane'
    var_16 = 25
    var_17 = False
    var_18 = 'Bob'
    var_19 = {var_0: var_18}
    var_20 = var_7.serialize(var_19)
    var_21 = 'Alice'
    var_22 = 'id'
    var_23 = module_0.String()
    var_24 = module_0.Integer()
    var_25 = {var_0: var_23, var_22: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'Charlie'
    var_28 = 123
    var_29 = {var_0: var_27, var_22: var_28}
    var_30 = var_26.serialize(var_29)
    var_31 = 'username'
    var_32 = 'details'
    var_33 = module_0.String()
    var_34 = 'email'
    var_35 = 'phone'
    var_36 = module_0.String()
    var_37 = module_0.String()
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {var_31: var_33, var_32: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = 'user1'
    var_43 = 'user@example.com'
    var_44 = '555-1234'
    var_45 = {var_34: var_43, var_35: var_44}
    var_46 = {var_31: var_42, var_32: var_45}
    var_47 = var_41.serialize(var_46)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_15.validate(var_19)
    var_21 = module_1.Field()
    var_22 = 'name'
    var_23 = {var_22: var_21}
    var_24 = module_0.Schema(var_23)
    var_25 = 1
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = 'invalid_key'
    var_30 = module_1.Field()
    var_31 = {var_22: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = 'required'
    var_36 = module_1.Field()
    var_37 = {var_22: var_36}
    var_38 = module_0.Schema(var_37)
    var_39 = 'John'
    var_40 = {var_22: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = module_1.Field(read_only=var_34)
    var_43 = module_1.Field()
    var_44 = 'id'
    var_45 = {var_44: var_42, var_22: var_43}
    var_46 = module_0.Schema(var_45)
    var_47 = 123
    var_48 = {var_22: var_39, var_44: var_47}
    var_49 = var_46.validate(var_48)
    var_50 = 'default_value'
    var_51 = module_1.Field(default=var_50)
    var_52 = 'status'
    var_53 = {var_52: var_51}
    var_54 = module_0.Schema(var_53)
    var_55 = {}
    var_56 = var_54.validate(var_55)
    var_57 = module_1.Field()
    var_58 = 'nested'
    var_59 = {var_58: var_57}
    var_60 = module_0.Schema(var_59)
    var_61 = 'nested'
    var_62 = 'value'
    var_63 = {var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = {}
    var_66 = module_0.Schema(var_65)
    var_67 = {}
    var_68 = var_66.validate(var_67)
    var_69 = module_1.Field()
    var_70 = {var_22: var_69}
    var_71 = module_0.Schema(var_70)
    var_72 = (var_22, var_39)
    var_73 = [var_72]
    var_74 = module_1.Field()
    var_75 = module_1.Field()
    var_76 = 'age'
    var_77 = {var_22: var_74, var_76: var_75}
    var_78 = module_0.Schema(var_77)
    var_79 = 30
    var_80 = {var_22: var_39, var_76: var_79}
    var_81 = var_78.validate(var_80)
    var_82 = module_1.Field()
    var_83 = {var_22: var_82}
    var_84 = module_0.Schema(var_83)
    var_85 = 'extra'
    var_86 = 'field'
    var_87 = {var_22: var_39, var_85: var_86}
    var_88 = var_84.validate(var_87)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'key2'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'value1'
    var_12 = 'value2'
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_1: var_15, var_8: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = var_18.serialize(var_13)
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_1: var_20, var_8: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_1: var_11}
    var_25 = var_23.serialize(var_24)
    var_26 = module_0.Field()
    var_27 = module_0.Field()
    var_28 = {var_1: var_26, var_8: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = var_29.serialize(var_24)
    var_31 = module_0.Field()
    var_32 = module_0.Field()
    var_33 = 'outer'
    var_34 = 'inner'
    var_35 = {var_33: var_32, var_34: var_31}
    var_36 = module_1.Schema(var_35)
    var_37 = 'outer_value'
    var_38 = 'inner_value'
    var_39 = {var_33: var_37, var_34: var_38}
    var_40 = var_36.serialize(var_39)
    var_41 = {}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_1: var_11}
    var_44 = var_42.serialize(var_43)
    var_45 = module_0.Field()
    var_46 = module_0.Field()
    var_47 = {var_1: var_45, var_8: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = (var_1, var_11)
    var_50 = (var_8, var_12)
    var_51 = [var_49, var_50]
    var_52 = var_48.serialize(var_43)



# Parsed testcases at query #8
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
    var_6 = None
    var_7 = var_5.serialize(var_6)
    assert var_7 is None
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_5.serialize(var_10)
    var_12 = 'Jane'
    var_13 = 25
    var_14 = 'Bob'
    var_15 = {var_0: var_14}
    var_16 = var_5.serialize(var_15)
    var_17 = 'Alice'
    var_18 = 'id'
    var_19 = module_0.String()
    var_20 = True
    var_21 = module_0.Integer()
    var_22 = {var_0: var_19, var_18: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Charlie'
    var_25 = {var_0: var_24, var_18: var_20}
    var_26 = var_23.serialize(var_25)
    var_27 = 'active'
    var_28 = module_0.String()
    var_29 = module_0.Boolean()
    var_30 = {var_0: var_28, var_27: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'David'
    var_33 = {var_0: var_32, var_27: var_20}
    var_34 = var_31.serialize(var_33)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test Schema.serialize() method with various inputs.'
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.serialize(var_5)
    assert var_6 is None
    var_7 = 'age'
    var_8 = module_0.Field()
    var_9 = module_0.Field()
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_2: var_12, var_7: var_13}
    var_15 = var_11.serialize(var_14)
    var_16 = 'Jane'
    var_17 = 25
    var_18 = 'email'
    var_19 = module_0.Field()
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_2: var_19, var_7: var_20, var_18: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Bob'
    var_25 = 35
    var_26 = {var_2: var_24, var_7: var_25}
    var_27 = var_23.serialize(var_26)
    var_28 = 'Alice'
    var_29 = 28
    var_30 = module_0.Field()
    var_31 = {var_2: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = {}
    var_34 = var_32.serialize(var_33)
    var_35 = module_0.Field()
    var_36 = {var_2: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'test'
    var_39 = {var_2: var_38}
    var_40 = var_37.serialize(var_39)
    var_41 = True
    var_42 = module_0.Field(read_only=var_41)
    var_43 = module_0.Field()
    var_44 = 'id'
    var_45 = {var_44: var_42, var_2: var_43}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_44: var_41, var_2: var_12}
    var_48 = var_46.serialize(var_47)
    var_49 = module_0.Field()
    var_50 = module_0.Field()
    var_51 = {var_2: var_49, var_7: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_2: var_12}
    var_54 = var_52.serialize(var_53)
    var_55 = module_0.Field()
    var_56 = module_0.Field()
    var_57 = module_0.Field()
    var_58 = {var_2: var_55, var_7: var_56, var_18: var_57}
    var_59 = module_1.Schema(var_58)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.Field()
    var_4 = module_1.Field()
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
    var_17 = None
    var_18 = var_8.validate(var_17)
    var_19 = None
    var_20 = False
    var_21 = module_0.Reference(var_7, var_0)
    var_22 = var_21.validate(var_19)
    var_23 = 'id'
    var_24 = 'data'
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'ComplexType'
    var_30 = module_0.Reference(var_29, var_0)
    var_31 = 'test'
    var_32 = {var_23: var_13, var_24: var_31}
    var_33 = var_30.validate(var_32)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = None
    var_19 = var_6.validate(var_18)
    var_20 = module_0.Reference(var_5, var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'John'
    var_15 = {var_12: var_14}
    var_16 = var_7.validate(var_15)
    var_17 = module_0.Definitions()
    var_18 = 'age'
    var_19 = module_1.Field()
    var_20 = {var_18: var_19}
    var_21 = module_0.Schema(var_20)
    var_22 = 'AgeSchema'
    var_23 = module_0.Reference(var_22, var_17)
    var_24 = 25
    var_25 = {var_18: var_24}
    var_26 = var_23.validate(var_25)
    var_27 = module_0.Definitions()
    var_28 = {}
    var_29 = module_0.Schema(var_28)
    var_30 = 'EmptySchema'
    var_31 = module_0.Reference(var_30, var_27)
    var_32 = 'not a dict'
    var_33 = var_31.validate(var_32)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = exc_info.value.messages()[var_6]
    var_11 = var_10.code
    assert var_11 == 'null'
    var_12 = {}
    var_13 = module_0.Schema(var_12)
    var_14 = 'invalid'
    var_15 = var_13.validate(var_14)
    var_16 = exc_info.value.messages()[var_6]
    var_17 = var_16.code
    assert var_17 == 'type'
    var_18 = {}
    var_19 = module_0.Schema(var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = {}
    var_23 = module_0.Schema(var_22)
    var_24 = 1
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)
    var_28 = 'invalid_key'
    var_29 = module_1.Field()
    var_30 = 'name'
    var_31 = {var_30: var_29}
    var_32 = module_0.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = 'required'
    var_36 = module_1.Field()
    var_37 = {var_30: var_36}
    var_38 = module_0.Schema(var_37)
    var_39 = 'John'
    var_40 = {var_30: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = module_1.Field(read_only=var_34)
    var_43 = 'id'
    var_44 = {var_43: var_42}
    var_45 = module_0.Schema(var_44)
    var_46 = 123
    var_47 = {var_43: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = 'default_value'
    var_50 = module_1.Field(default=var_49)
    var_51 = 'status'
    var_52 = {var_51: var_50}
    var_53 = module_0.Schema(var_52)
    var_54 = {}
    var_55 = var_53.validate(var_54)
    var_56 = 'age'
    var_57 = 5
    var_58 = module_1.String(max_length=var_57)
    var_59 = module_1.Integer()
    var_60 = {var_30: var_58, var_56: var_59}
    var_61 = module_0.Schema(var_60)
    var_62 = 'name'
    var_63 = 'age'
    var_64 = 'VeryLongName'
    var_65 = 'not_an_int'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_61.validate(var_66)
    var_68 = 'key'
    var_69 = module_1.Field()
    var_70 = {var_68: var_69}
    var_71 = module_0.Schema(var_70)
    var_72 = 'value'
    var_73 = {var_68: var_72}
    var_74 = var_71.validate(var_73)
    var_75 = module_1.Field()
    var_76 = {var_30: var_75}
    var_77 = module_0.Schema(var_76)
    var_78 = 'extra'
    var_79 = 'field'
    var_80 = {var_30: var_39, var_78: var_79}
    var_81 = var_77.validate(var_80)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'key2'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'value1'
    var_12 = 'value2'
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_1: var_15, var_8: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = var_18.serialize(var_13)
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_1: var_20, var_8: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_1: var_11}
    var_25 = var_23.serialize(var_24)
    var_26 = module_0.Field()
    var_27 = {var_1: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'extra'
    var_30 = 'extra_value'
    var_31 = {var_1: var_11, var_29: var_30}
    var_32 = var_28.serialize(var_31)
    var_33 = module_0.Field()
    var_34 = module_0.Field()
    var_35 = {var_1: var_33, var_8: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = var_36.serialize(var_31)
    var_38 = module_0.Field()
    var_39 = {var_1: var_11, var_8: var_12}
    var_40 = var_36.serialize(var_39)
    var_41 = {}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_1: var_11}
    var_44 = var_42.serialize(var_43)
    var_45 = module_0.Field()
    var_46 = module_0.Field()
    var_47 = {var_1: var_45, var_8: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_1: var_4, var_8: var_12}
    var_50 = var_48.serialize(var_49)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None
    var_6 = module_0.Field()
    var_7 = module_0.Field()
    var_8 = 'key2'
    var_9 = {var_1: var_6, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'value1'
    var_12 = 'value2'
    var_13 = {var_1: var_11, var_8: var_12}
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_1: var_15, var_8: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = var_18.serialize(var_13)
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_1: var_20, var_8: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_1: var_11}
    var_25 = var_23.serialize(var_24)
    var_26 = module_0.Field()
    var_27 = module_0.Field()
    var_28 = {var_1: var_26, var_8: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = var_29.serialize(var_24)
    var_31 = module_0.Field()
    var_32 = {var_1: var_26, var_8: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {var_1: var_11, var_8: var_12}
    var_35 = var_33.serialize(var_34)
    var_36 = {}
    var_37 = module_1.Schema(var_36)
    var_38 = {var_1: var_11}
    var_39 = var_37.serialize(var_38)
    var_40 = module_0.Field()
    var_41 = {var_1: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_1: var_11, var_8: var_12}
    var_44 = var_42.serialize(var_43)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Test Reference.validate() method.'
    var_1 = module_0.Definitions()
    var_2 = 'name'
    var_3 = module_1.Field()
    var_4 = {var_2: var_3}
    var_5 = module_0.Schema(var_4)
    var_6 = 'TestSchema'
    var_7 = module_0.Reference(var_6, var_1)
    var_8 = 'test'
    var_9 = {var_2: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = True
    var_12 = module_0.Reference(var_6, var_1)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = False
    var_16 = module_0.Reference(var_6, var_1)
    var_17 = None
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Reference(var_6, var_1)
    var_20 = 'not a dict'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Reference(var_6, var_1)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = module_0.Definitions()
    var_26 = 'id'
    var_27 = 'value'
    var_28 = module_1.Field()
    var_29 = module_1.Field()
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = module_0.Schema(var_30)
    var_32 = 'ComplexSchema'
    var_33 = module_0.Reference(var_32, var_25)
    var_34 = {var_26: var_11, var_27: var_8}
    var_35 = var_33.validate(var_34)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = 'id'
    var_15 = module_1.Field()
    var_16 = module_1.Field()
    var_17 = {var_14: var_15, var_11: var_16}
    var_18 = module_0.Schema(var_17)
    var_19 = 'Person'
    var_20 = module_0.Reference(var_19, var_13)
    var_21 = 'John'
    var_22 = {var_14: var_6, var_11: var_21}
    var_23 = var_20.validate(var_22)
    var_24 = module_0.Definitions()
    var_25 = 'email'
    var_26 = module_1.Field()
    var_27 = {var_25: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'User'
    var_30 = module_0.Reference(var_29, var_24)
    var_31 = None
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Definitions()
    var_34 = module_1.Field()
    var_35 = 'required_field'
    var_36 = {var_35: var_34}
    var_37 = module_0.Schema(var_36)
    var_38 = 'RequiredSchema'
    var_39 = module_0.Reference(var_38, var_33)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Definitions()
    var_43 = 'value'
    var_44 = module_1.Field()
    var_45 = {var_43: var_44}
    var_46 = module_0.Schema(var_45)
    var_47 = 'Inner'
    var_48 = module_0.Reference(var_47, var_42)
    var_49 = 'test'
    var_50 = {var_43: var_49}
    var_51 = var_48.validate(var_50)



# Parsed testcases at query #18
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
    var_10 = True
    var_11 = module_1.Schema(var_4)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = None
    var_15 = var_5.validate(var_14)
    var_16 = 'invalid'
    var_17 = var_5.validate(var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = var_5.validate(var_21)
    var_23 = 1
    var_24 = 'name'
    var_25 = 'age'
    var_26 = 'value'
    var_27 = 'John'
    var_28 = 30
    var_29 = {var_23: var_26, var_24: var_27, var_25: var_28}
    var_30 = var_5.validate(var_29)
    var_31 = 'invalid_key'
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = {var_23: var_32, var_24: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = 'name'
    var_37 = 'John'
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = 'required'
    var_41 = 'status'
    var_42 = module_0.String()
    var_43 = 'active'
    var_44 = module_0.String()
    var_45 = {var_36: var_42, var_41: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_36: var_27}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.String()
    var_51 = module_0.Integer()
    var_52 = {var_36: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = 123
    var_55 = {var_36: var_27, var_49: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_36: var_57, var_37: var_58}
    var_60 = module_1.Schema(var_59)
    var_61 = 'name'
    var_62 = 'age'
    var_63 = 'John'
    var_64 = 'not_an_int'
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = var_60.validate(var_65)
    var_67 = {var_61: var_65, var_62: var_66}
    var_68 = 'extra'
    var_69 = 'field'
    var_70 = {var_61: var_65, var_62: var_66, var_68: var_69}
    var_71 = var_5.validate(var_70)
    var_72 = 1
    var_73 = 'age'
    var_74 = 'value'
    var_75 = 'invalid'
    var_76 = {var_72: var_74, var_73: var_75}
    var_77 = var_35.validate(var_76)
    var_78 = 'Unknown'
    var_79 = module_0.String()
    var_80 = 0
    var_81 = module_0.Integer()
    var_82 = {var_72: var_79, var_73: var_81}
    var_83 = module_1.Schema(var_82)
    var_84 = {}
    var_85 = var_83.validate(var_84)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Unit tests for Reference.validate() method.'
    var_1 = module_0.Definitions()
    var_2 = 'name'
    var_3 = module_1.Field()
    var_4 = {var_2: var_3}
    var_5 = module_0.Schema(var_4)
    var_6 = 'TestSchema'
    var_7 = True
    var_8 = module_0.Reference(var_6, var_1)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Reference(var_6, var_1)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Reference(var_6, var_1)
    var_16 = 'test'
    var_17 = {var_14: var_16}
    var_18 = var_15.validate(var_17)
    var_19 = module_0.Reference(var_6, var_1)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Reference(var_6, var_1)
    var_23 = 'id'
    var_24 = 'data'
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'NestedSchema'
    var_30 = module_0.Reference(var_29, var_1)
    var_31 = 'value'
    var_32 = {var_23: var_7, var_24: var_31}
    var_33 = var_30.validate(var_32)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'John'
    var_14 = {var_11: var_13}
    var_15 = var_10.validate(var_14)
    var_16 = 'required_field'
    var_17 = False
    var_18 = module_1.Field(allow_null=var_17)
    var_19 = {var_16: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'StrictSchema'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = 'age'
    var_26 = module_1.Integer()
    var_27 = {var_25: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'IntSchema'
    var_30 = module_0.Reference(var_29, var_0)
    var_31 = 25
    var_32 = {var_25: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = 'age'
    var_35 = 'not_an_int'
    var_36 = {var_34: var_35}
    var_37 = var_30.validate(var_36)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Reference(var_5, var_0)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = 'id'
    var_22 = 'data'
    var_23 = module_1.Field()
    var_24 = module_1.Field()
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'ComplexSchema'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = {var_21: var_10, var_22: var_7}
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Definitions()
    var_32 = 'field1'
    var_33 = module_1.Field()
    var_34 = {var_32: var_33}
    var_35 = module_0.Schema(var_34)
    var_36 = 'MockSchema'
    var_37 = module_0.Reference(var_36, var_31)
    var_38 = 'value1'
    var_39 = {var_32: var_38}
    var_40 = var_37.validate(var_39)



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'test_schema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Reference(var_5, var_0)
    var_15 = 'test'
    var_16 = {var_12: var_15}
    var_17 = var_14.validate(var_16)
    var_18 = module_1.Field()
    var_19 = {var_12: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'invalid_schema'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = False
    var_26 = var_4.validate
    var_27 = module_0.Reference(var_5, var_0)
    var_28 = 'value'
    var_29 = {var_23: var_28}
    var_30 = var_27.validate(var_29)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Reference(var_5, var_0)
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = {}
    var_22 = var_6.validate(var_21)



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'test'
    var_15 = {var_12: var_14}
    var_16 = var_7.validate(var_15)
    var_17 = 'invalid'
    var_18 = var_7.validate(var_17)
    var_19 = 'id'
    var_20 = 'value'
    var_21 = module_1.Field()
    var_22 = module_1.Field()
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.Schema(var_23)
    var_25 = 'NestedSchema'
    var_26 = module_0.Reference(var_25, var_0)
    var_27 = {var_19: var_6, var_20: var_14}
    var_28 = var_26.validate(var_27)



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = module_1.Field()
    var_15 = 'SimpleField'
    var_16 = module_0.Reference(var_15, var_13)
    var_17 = 'test_value'
    var_18 = var_16.validate(var_17)
    assert var_18 == 'test_value'
    var_19 = module_0.Definitions()
    var_20 = module_1.Field()
    var_21 = {var_11: var_20}
    var_22 = module_0.Schema(var_21)
    var_23 = 'PersonSchema'
    var_24 = module_0.Reference(var_23, var_19)
    var_25 = 'John'
    var_26 = {var_11: var_25}
    var_27 = var_24.validate(var_26)
    var_28 = module_0.Definitions()
    var_29 = 'age'
    var_30 = module_1.Field()
    var_31 = {var_29: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'StrictSchema'
    var_34 = module_0.Reference(var_33, var_28)
    var_35 = None
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Definitions()
    var_38 = 'id'
    var_39 = module_1.Field()
    var_40 = {var_38: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = 'TargetRef'
    var_43 = module_0.Reference(var_42, var_37)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = 'John'
    var_15 = {var_11: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = 'age'
    var_18 = module_1.Field()
    var_19 = {var_17: var_18}
    var_20 = module_0.Schema(var_19)
    var_21 = 'InnerSchema'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 25
    var_24 = {var_17: var_23}
    var_25 = var_22.validate(var_24)
    var_26 = module_0.Definitions()
    var_27 = 'id'
    var_28 = module_1.Field()
    var_29 = {var_27: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'StrictSchema'
    var_32 = module_0.Reference(var_31, var_26)
    var_33 = 123
    var_34 = {var_27: var_33}
    var_35 = var_32.validate(var_34)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = module_1.Field()
    var_15 = {var_11: var_14}
    var_16 = module_0.Schema(var_15)
    var_17 = 'TestSchema2'
    var_18 = module_0.Reference(var_17, var_13)
    var_19 = 'test'
    var_20 = {var_11: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = module_0.Definitions()
    var_23 = False
    var_24 = module_1.Field(allow_null=var_23)
    var_25 = {var_11: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'TestSchema3'
    var_28 = module_0.Reference(var_27, var_22)
    var_29 = 'name'
    var_30 = None
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Definitions()
    var_34 = 'id'
    var_35 = 'value'
    var_36 = module_1.Field()
    var_37 = module_1.Field()
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = module_0.Schema(var_38)
    var_40 = 'ComplexSchema'
    var_41 = module_0.Reference(var_40, var_33)
    var_42 = {var_34: var_6, var_35: var_19}
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Test Reference.validate method'
    var_1 = module_0.Definitions()
    var_2 = 'name'
    var_3 = module_1.Field()
    var_4 = {var_2: var_3}
    var_5 = module_0.Schema(var_4)
    var_6 = 'TestSchema'
    var_7 = module_0.Reference(var_6, var_1)
    var_8 = 'test'
    var_9 = {var_2: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = False
    var_12 = module_0.Reference(var_6, var_1)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = True
    var_16 = module_0.Reference(var_6, var_1)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = module_0.Reference(var_6, var_1)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = 'id'
    var_23 = module_1.Field()
    var_24 = module_1.Field()
    var_25 = {var_22: var_23, var_21: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'ComplexSchema'
    var_28 = module_0.Reference(var_27, var_1)
    var_29 = 'example'
    var_30 = {var_22: var_15, var_21: var_29}
    var_31 = var_28.validate(var_30)



# Parsed testcases at query #29
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test_schema'
    var_5 = True
    var_6 = module_0.Reference(var_4, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = False
    var_10 = module_0.Reference(var_4, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'John'
    var_14 = {var_11: var_13}
    var_15 = var_6.validate(var_14)
    var_16 = 'age'
    var_17 = module_1.Field(allow_null=var_9)
    var_18 = {var_16: var_17}
    var_19 = module_0.Schema(var_18)
    var_20 = 'schema_with_field'
    var_21 = module_0.Reference(var_20, var_0)
    var_22 = {var_16: var_7}
    var_23 = var_21.validate(var_22)
    var_24 = {}
    var_25 = var_6.validate(var_24)
    var_26 = 'id'
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_26: var_27, var_23: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'nested'
    var_32 = module_0.Reference(var_31, var_0)
    var_33 = 123
    var_34 = 'Test'
    var_35 = {var_26: var_33, var_23: var_34}
    var_36 = var_32.validate(var_35)



# Parsed testcases at query #30
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = 'test'
    var_14 = {var_7: var_13}
    var_15 = var_6.validate(var_14)
    var_16 = 'invalid'
    var_17 = var_6.validate(var_16)
    var_18 = 'age'
    var_19 = module_1.Field()
    var_20 = {var_18: var_19}
    var_21 = module_0.Schema(var_20)
    var_22 = 'SchemaWithAge'
    var_23 = module_0.Reference(var_22, var_0)
    var_24 = 25
    var_25 = {var_18: var_24}
    var_26 = var_23.validate(var_25)



# Parsed testcases at query #31
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = var_11.validate(var_17)
    var_19 = {}
    var_20 = module_0.Schema(var_19)
    var_21 = 1
    var_22 = 'key'
    var_23 = 'value'
    var_24 = {var_21: var_23, var_22: var_23}
    var_25 = var_20.validate(var_24)
    var_26 = 'invalid_key'
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = 'required_field'
    var_30 = 'optional_field'
    var_31 = {var_29: var_27, var_30: var_28}
    var_32 = module_0.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = 'required'
    var_36 = module_1.Field()
    var_37 = module_1.Field()
    var_38 = 'field1'
    var_39 = 'field2'
    var_40 = {var_38: var_36, var_39: var_37}
    var_41 = module_0.Schema(var_40)
    var_42 = 'value1'
    var_43 = 'value2'
    var_44 = {var_38: var_42, var_39: var_43}
    var_45 = var_41.validate(var_44)
    var_46 = module_1.Field(read_only=var_34)
    var_47 = module_1.Field()
    var_48 = 'read_only_field'
    var_49 = {var_48: var_46, var_39: var_47}
    var_50 = module_0.Schema(var_49)
    var_51 = 'ignored'
    var_52 = {var_48: var_51, var_39: var_43}
    var_53 = var_50.validate(var_52)
    var_54 = 'default_value'
    var_55 = module_1.Field(default=var_54)
    var_56 = module_1.Field()
    var_57 = 'field_with_default'
    var_58 = {var_57: var_55, var_39: var_56}
    var_59 = module_0.Schema(var_58)
    var_60 = {var_39: var_43}
    var_61 = var_59.validate(var_60)
    var_62 = module_1.Field()
    var_63 = {var_38: var_62}
    var_64 = module_0.Schema(var_63)
    var_65 = (var_38, var_42)
    var_66 = [var_65]
    var_67 = module_1.Field()
    var_68 = ()
    var_69 = 'child_error'
    var_70 = module_2.ValidationError(code=var_69)
    var_71 = 'child'
    var_72 = {var_71: var_67}
    var_73 = module_0.Schema(var_72)
    var_74 = 'child'
    var_75 = 'invalid'
    var_76 = {var_74: var_75}
    var_77 = var_73.validate(var_76)
    var_78 = module_1.Field()
    var_79 = module_1.Field()
    var_80 = {var_38: var_78, var_39: var_79}
    var_81 = module_0.Schema(var_80)
    var_82 = 1
    var_83 = 'extra'
    var_84 = 'value'
    var_85 = 'data'
    var_86 = {var_82: var_84, var_83: var_85}
    var_87 = var_81.validate(var_86)
    var_88 = {}
    var_89 = module_0.Schema(var_88)
    var_90 = {}
    var_91 = var_89.validate(var_90)
    var_92 = module_1.Field()
    var_93 = {var_38: var_92}
    var_94 = module_0.Schema(var_93)
    var_95 = 'extra_key'
    var_96 = 'extra_value'
    var_97 = {var_38: var_42, var_95: var_96}
    var_98 = var_94.validate(var_97)



# Parsed testcases at query #32
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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = {var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.String()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 1
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'invalid_key'
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_29: var_34, var_30: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'required'
    var_43 = module_0.String()
    var_44 = 0
    var_45 = module_0.Integer()
    var_46 = {var_38: var_43, var_39: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {var_38: var_6}
    var_49 = var_47.validate(var_48)
    var_50 = 'id'
    var_51 = module_0.String()
    var_52 = module_0.String()
    var_53 = {var_38: var_51, var_50: var_52}
    var_54 = module_1.Schema(var_53)
    var_55 = '123'
    var_56 = {var_38: var_6, var_50: var_55}
    var_57 = var_54.validate(var_56)
    var_58 = module_0.String()
    var_59 = module_0.Integer()
    var_60 = {var_38: var_58, var_39: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = 'name'
    var_63 = 'age'
    var_64 = 'John'
    var_65 = 'not_an_int'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_61.validate(var_66)
    var_68 = {}
    var_69 = module_1.Schema(var_68)
    var_70 = {}
    var_71 = var_69.validate(var_70)
    var_72 = module_0.String()
    var_73 = {var_62: var_72}
    var_74 = module_1.Schema(var_73)
    var_75 = 'extra'
    var_76 = 'field'
    var_77 = {var_62: var_67, var_75: var_76}
    var_78 = var_74.validate(var_77)
    var_79 = module_0.String()
    var_80 = {var_62: var_79}
    var_81 = module_1.Schema(var_80)
    var_82 = (var_62, var_67)
    var_83 = [var_82]



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = {var_1: var_0}
    var_8 = False
    var_9 = module_1.Schema(var_7)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = {var_10: var_0}
    var_13 = module_1.Schema(var_12)
    var_14 = 'invalid'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Field(allow_null=var_3)
    var_17 = {var_14: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'test'
    var_20 = {var_14: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = module_0.Field(allow_null=var_8)
    var_23 = {var_14: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'required'
    var_28 = module_0.Field()
    var_29 = {var_25: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 123
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'invalid_key'
    var_36 = 'default_name'
    var_37 = module_0.Field(default=var_36)
    var_38 = {var_31: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field(read_only=var_33)
    var_43 = module_0.Field()
    var_44 = 'id'
    var_45 = {var_44: var_42, var_31: var_43}
    var_46 = module_1.Schema(var_45)
    var_47 = 123
    var_48 = {var_44: var_47, var_31: var_19}
    var_49 = var_46.validate(var_48)
    var_50 = module_0.Field()
    var_51 = {var_31: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_31: var_19}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Field()
    var_56 = {var_31: var_55}
    var_57 = module_1.Schema(var_56)
    var_58 = 'name'
    var_59 = 'test'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)
    var_62 = module_0.Field(allow_null=var_60)
    var_63 = module_0.Field(allow_null=var_60)
    var_64 = 'age'
    var_65 = {var_58: var_62, var_64: var_63}
    var_66 = module_1.Schema(var_65)
    var_67 = 'John'
    var_68 = 30
    var_69 = {var_58: var_67, var_64: var_68}
    var_70 = var_66.validate(var_69)
    var_71 = module_0.Field(allow_null=var_60)
    var_72 = {var_58: var_71}
    var_73 = module_1.Schema(var_72)
    var_74 = 'extra'
    var_75 = 'field'
    var_76 = {var_58: var_19, var_74: var_75}
    var_77 = var_73.validate(var_76)



# Parsed testcases at query #34
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = True
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Reference(var_5, var_0)
    var_17 = 'valid'
    var_18 = {var_10: var_17}
    var_19 = var_16.validate(var_18)
    var_20 = 'required_field'
    var_21 = False
    var_22 = module_1.Field(allow_null=var_21)
    var_23 = {var_20: var_22}
    var_24 = module_0.Schema(var_23)
    var_25 = 'InvalidSchema'
    var_26 = module_0.Reference(var_25, var_0)
    var_27 = {}
    var_28 = var_26.validate(var_27)



# Parsed testcases at query #35
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Reference(var_5, var_0)
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = 'id'
    var_22 = 'data'
    var_23 = module_1.Field()
    var_24 = module_1.Field()
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'NestedSchema'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = 'value'
    var_30 = {var_21: var_10, var_22: var_29}
    var_31 = var_28.validate(var_30)



# Parsed testcases at query #36
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = 'id'
    var_15 = module_1.Field()
    var_16 = module_1.Field()
    var_17 = {var_14: var_15, var_11: var_16}
    var_18 = module_0.Schema(var_17)
    var_19 = 'User'
    var_20 = module_0.Reference(var_19, var_13)
    var_21 = 'John'
    var_22 = {var_14: var_6, var_11: var_21}
    var_23 = var_20.validate(var_22)
    var_24 = module_0.Definitions()
    var_25 = 'age'
    var_26 = False
    var_27 = module_1.Integer()
    var_28 = {var_25: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = 'Person'
    var_31 = module_0.Reference(var_30, var_24)
    var_32 = 'age'
    var_33 = None
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = module_0.Definitions()
    var_37 = 'optional_field'
    var_38 = module_1.Field()
    var_39 = {var_37: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = 'EmptySchema'
    var_42 = module_0.Reference(var_41, var_36)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Definitions()
    var_46 = 'value'
    var_47 = module_1.Field()
    var_48 = {var_46: var_47}
    var_49 = module_0.Schema(var_48)
    var_50 = 'ValueSchema'
    var_51 = module_0.Reference(var_50, var_45)
    var_52 = 'test'
    var_53 = {var_46: var_52}
    var_54 = var_51.validate(var_53)



# Parsed testcases at query #37
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = module_1.Field()
    var_15 = {var_11: var_14}
    var_16 = module_0.Schema(var_15)
    var_17 = 'TestSchema2'
    var_18 = module_0.Reference(var_17, var_13)
    var_19 = 'test'
    var_20 = {var_11: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = module_0.Definitions()
    var_23 = 'email'
    var_24 = module_1.String()
    var_25 = {var_23: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'UserSchema'
    var_28 = module_0.Reference(var_27, var_22)
    var_29 = 'test@example.com'
    var_30 = {var_23: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = 'age'
    var_34 = module_1.Field()
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'PersonSchema'
    var_38 = module_0.Reference(var_37, var_32)
    var_39 = 25
    var_40 = {var_33: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = module_0.Definitions()
    var_43 = 'id'
    var_44 = module_1.Field()
    var_45 = {var_43: var_44}
    var_46 = module_0.Schema(var_45)
    var_47 = 'ItemSchema'
    var_48 = module_0.Reference(var_47, var_42)



# Parsed testcases at query #38
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = True
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Reference(var_5, var_0)
    var_17 = 'another_test'
    var_18 = {var_10: var_17}
    var_19 = var_16.validate(var_18)



# Parsed testcases at query #39
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'name'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = {var_21: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'test'
    var_30 = {var_21: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_1.Field(read_only=var_25)
    var_33 = 'id'
    var_34 = {var_33: var_32}
    var_35 = module_0.Schema(var_34)
    var_36 = '123'
    var_37 = {var_33: var_36}
    var_38 = var_35.validate(var_37)
    var_39 = 'default_value'
    var_40 = module_1.Field(default=var_39)
    var_41 = 'status'
    var_42 = {var_41: var_40}
    var_43 = module_0.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)
    var_46 = module_1.Field()
    var_47 = ()
    var_48 = 'Invalid'
    var_49 = 'invalid'
    var_50 = []
    var_51 = module_2.Message(text=var_48, code=var_49, index=var_50)
    var_52 = [var_51]
    var_53 = module_2.ValidationError(messages=var_52)
    var_54 = 'nested'
    var_55 = {var_54: var_46}
    var_56 = module_0.Schema(var_55)
    var_57 = 'nested'
    var_58 = 'value'
    var_59 = {var_57: var_58}
    var_60 = var_56.validate(var_59)
    var_61 = 'key'
    var_62 = module_1.Field()
    var_63 = {var_61: var_62}
    var_64 = module_0.Schema(var_63)
    var_65 = 'value'
    var_66 = (var_61, var_65)
    var_67 = [var_66]
    var_68 = module_1.Field()
    var_69 = module_1.Field()
    var_70 = 'field1'
    var_71 = 'field2'
    var_72 = {var_70: var_68, var_71: var_69}
    var_73 = module_0.Schema(var_72)
    var_74 = 'value1'
    var_75 = 'value2'
    var_76 = {var_70: var_74, var_71: var_75}
    var_77 = var_73.validate(var_76)
    var_78 = {}
    var_79 = module_0.Schema(var_78)
    var_80 = {}
    var_81 = var_79.validate(var_80)



# Parsed testcases at query #40
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'TestSchema'
    var_5 = True
    var_6 = module_0.Reference(var_4, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = module_0.Reference(var_4, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = 'age'
    var_13 = module_1.Field()
    var_14 = module_1.Field()
    var_15 = {var_10: var_13, var_12: var_14}
    var_16 = module_0.Schema(var_15)
    var_17 = 'PersonSchema'
    var_18 = module_0.Reference(var_17, var_0)
    var_19 = 'John'
    var_20 = 30
    var_21 = {var_10: var_19, var_12: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = 'StringField'
    var_24 = module_0.Reference(var_23, var_0)
    var_25 = 'test_value'
    var_26 = var_24.validate(var_25)
    assert var_26 == 'test_value'
    var_27 = 'email'
    var_28 = module_1.Field()
    var_29 = {var_27: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = 'EmailSchema'
    var_32 = module_0.Reference(var_31, var_0)
    var_33 = 'test@example.com'
    var_34 = {var_27: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_0.Reference(var_4, var_0)
    var_37 = 'not_a_dict'
    var_38 = var_36.validate(var_37)



# Parsed testcases at query #41
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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = {var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.String()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 123
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'invalid_key'
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_29: var_34, var_30: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'required'
    var_43 = 'id'
    var_44 = module_0.String()
    var_45 = module_0.String()
    var_46 = {var_38: var_44, var_43: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = '123'
    var_49 = {var_38: var_6, var_43: var_48}
    var_50 = var_47.validate(var_49)
    var_51 = 'status'
    var_52 = module_0.String()
    var_53 = 'active'
    var_54 = module_0.String()
    var_55 = {var_38: var_52, var_51: var_54}
    var_56 = module_1.Schema(var_55)
    var_57 = {var_38: var_6}
    var_58 = var_56.validate(var_57)
    var_59 = module_0.String()
    var_60 = module_0.Integer()
    var_61 = {var_38: var_59, var_39: var_60}
    var_62 = module_1.Schema(var_61)
    var_63 = 'name'
    var_64 = 'age'
    var_65 = 'John'
    var_66 = 'not an integer'
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = var_62.validate(var_67)
    var_69 = [var_64]
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = {var_63: var_70, var_64: var_71}
    var_73 = module_1.Schema(var_72)
    var_74 = 123
    var_75 = 'age'
    var_76 = 'value'
    var_77 = 'not an integer'
    var_78 = {var_74: var_76, var_75: var_77}
    var_79 = var_73.validate(var_78)
    var_80 = module_0.String()
    var_81 = {var_74: var_80}
    var_82 = module_1.Schema(var_81)
    var_83 = (var_74, var_79)
    var_84 = [var_83]
    var_85 = module_0.String()
    var_86 = {var_74: var_85}
    var_87 = module_1.Schema(var_86)
    var_88 = 'extra'
    var_89 = 'field'
    var_90 = {var_74: var_79, var_88: var_89}
    var_91 = var_87.validate(var_90)



# Parsed testcases at query #42
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Definitions()
    var_14 = module_1.Field()
    var_15 = {var_11: var_14}
    var_16 = module_0.Schema(var_15)
    var_17 = 'Person'
    var_18 = module_0.Reference(var_17, var_13)
    var_19 = 'John'
    var_20 = {var_11: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = module_0.Definitions()
    var_23 = False
    var_24 = module_1.Field(allow_null=var_23)
    var_25 = 'email'
    var_26 = {var_25: var_24}
    var_27 = module_0.Schema(var_26)
    var_28 = 'User'
    var_29 = module_0.Reference(var_28, var_22)
    var_30 = {var_25: var_8}
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = module_1.Field(allow_null=var_6)
    var_34 = 'optional'
    var_35 = {var_34: var_33}
    var_36 = module_0.Schema(var_35)
    var_37 = 'Optional'
    var_38 = module_0.Reference(var_37, var_32)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Definitions()
    var_42 = 'id'
    var_43 = module_1.Field()
    var_44 = {var_42: var_43}
    var_45 = module_0.Schema(var_44)
    var_46 = 'TargetType'
    var_47 = module_0.Reference(var_46, var_41)



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = {var_0: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.String()
    var_13 = {var_10: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = 'not a dict'
    var_16 = var_14.validate(var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = var_14.validate(var_20)
    var_22 = 'age'
    var_23 = module_0.String()
    var_24 = module_0.Integer()
    var_25 = {var_17: var_23, var_22: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'John'
    var_28 = 30
    var_29 = {var_17: var_27, var_22: var_28}
    var_30 = var_26.validate(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = {var_17: var_31, var_22: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = 'name'
    var_36 = 'John'
    var_37 = {var_35: var_36}
    var_38 = var_34.validate(var_37)
    var_39 = 'required'
    var_40 = module_0.String()
    var_41 = {var_35: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = 1
    var_44 = 'value'
    var_45 = {var_43: var_44}
    var_46 = var_42.validate(var_45)
    var_47 = 'invalid_key'
    var_48 = module_0.Integer()
    var_49 = {var_22: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = 'age'
    var_52 = 'not an integer'
    var_53 = {var_51: var_52}
    var_54 = var_50.validate(var_53)
    var_55 = module_0.String()
    var_56 = 'id'
    var_57 = module_0.String()
    var_58 = {var_51: var_57, var_56: var_55}
    var_59 = module_1.Schema(var_58)
    var_60 = '123'
    var_61 = {var_51: var_27, var_56: var_60}
    var_62 = var_59.validate(var_61)
    var_63 = 'status'
    var_64 = module_0.String()
    var_65 = 'active'
    var_66 = module_0.String()
    var_67 = {var_51: var_64, var_63: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = {var_51: var_27}
    var_70 = var_68.validate(var_69)
    var_71 = module_0.String()
    var_72 = {var_51: var_71}
    var_73 = module_1.Schema(var_72)
    var_74 = 'extra'
    var_75 = 'field'
    var_76 = {var_51: var_27, var_74: var_75}
    var_77 = var_73.validate(var_76)
    var_78 = 'Unknown'
    var_79 = module_0.String()
    var_80 = 0
    var_81 = module_0.Integer()
    var_82 = {var_51: var_79, var_22: var_81}
    var_83 = module_1.Schema(var_82)
    var_84 = {}
    var_85 = var_83.validate(var_84)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = module_0.Boolean()
    var_89 = {var_51: var_86, var_22: var_87, var_65: var_88}
    var_90 = module_1.Schema(var_89)
    var_91 = 'age'
    var_92 = 'active'
    var_93 = 'invalid'
    var_94 = {var_91: var_93, var_92: var_93}
    var_95 = var_90.validate(var_94)
    var_96 = module_0.String()
    var_97 = {var_91: var_96}
    var_98 = module_1.Schema(var_97)
    var_99 = (var_91, var_27)
    var_100 = [var_99]



# Parsed testcases at query #44
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'field1'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = {var_21: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'value'
    var_30 = {var_21: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_1.Field(read_only=var_25)
    var_33 = {var_21: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 'default_value'
    var_38 = module_1.Field(default=var_37)
    var_39 = {var_21: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = module_1.Field()
    var_44 = ()
    var_45 = 'error'
    var_46 = module_2.Message(text=var_45, code=var_45)
    var_47 = [var_46]
    var_48 = module_2.ValidationError(messages=var_47)
    var_49 = {var_21: var_43}
    var_50 = module_0.Schema(var_49)
    var_51 = 'field1'
    var_52 = 'value'
    var_53 = {var_51: var_52}
    var_54 = var_50.validate(var_53)
    var_55 = module_1.Field()
    var_56 = {var_21: var_55}
    var_57 = module_0.Schema(var_56)
    var_58 = (var_21, var_29)
    var_59 = [var_58]
    var_60 = module_1.Field()
    var_61 = module_1.Field()
    var_62 = 'field2'
    var_63 = {var_21: var_60, var_62: var_61}
    var_64 = module_0.Schema(var_63)
    var_65 = 'value1'
    var_66 = 'value2'
    var_67 = {var_21: var_65, var_62: var_66}
    var_68 = var_64.validate(var_67)
    var_69 = module_1.Field()
    var_70 = module_1.Field(read_only=var_52)
    var_71 = {var_21: var_69, var_62: var_70}
    var_72 = module_0.Schema(var_71)
    var_73 = 'should_be_ignored'
    var_74 = {var_21: var_65, var_62: var_73}
    var_75 = var_72.validate(var_74)



# Parsed testcases at query #45
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_6.validate(var_9)
    var_11 = 'test'
    var_12 = {var_9: var_11}
    var_13 = var_6.validate(var_12)
    var_14 = module_0.Reference(var_5, var_0)
    var_15 = 'another'
    var_16 = {var_9: var_15}
    var_17 = var_14.validate(var_16)



# Parsed testcases at query #46
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Reference(var_5, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Reference(var_5, var_0)
    var_14 = 'test'
    var_15 = {var_11: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = module_1.Field()
    var_18 = {var_11: var_17}
    var_19 = module_0.Schema(var_18)
    var_20 = module_0.Definitions()
    var_21 = 'NameSchema'
    var_22 = module_0.Reference(var_21, var_20)
    var_23 = 'John'
    var_24 = {var_11: var_23}
    var_25 = var_22.validate(var_24)
    var_26 = 'id'
    var_27 = module_1.String()
    var_28 = {var_26: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = module_0.Definitions()
    var_31 = 'StrictSchema'
    var_32 = module_0.Reference(var_31, var_30)
    var_33 = '123'
    var_34 = {var_26: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_0.Reference(var_5, var_0)



# Parsed testcases at query #47
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_1.Field()
    var_21 = 'name'
    var_22 = {var_21: var_20}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field(allow_null=var_25)
    var_27 = {var_21: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'John'
    var_30 = {var_21: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_1.Field(read_only=var_25)
    var_33 = module_1.Field()
    var_34 = 'id'
    var_35 = {var_34: var_32, var_21: var_33}
    var_36 = module_0.Schema(var_35)
    var_37 = {var_34: var_25, var_21: var_29}
    var_38 = var_36.validate(var_37)
    var_39 = 'default_name'
    var_40 = module_1.Field(default=var_39)
    var_41 = {var_21: var_40}
    var_42 = module_0.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = 'key'
    var_46 = 'value'
    var_47 = (var_45, var_46)
    var_48 = [var_47]
    var_49 = module_1.Field()
    var_50 = {var_45: var_49}
    var_51 = module_0.Schema(var_50)
    var_52 = module_1.Field()
    var_53 = {var_21: var_52}
    var_54 = module_0.Schema(var_53)
    var_55 = 'name'
    var_56 = None
    var_57 = {var_55: var_56}
    var_58 = var_54.validate(var_57)
    var_59 = {}
    var_60 = module_0.Schema(var_59)
    var_61 = {}
    var_62 = var_60.validate(var_61)
    var_63 = module_1.Field()
    var_64 = module_1.Field()
    var_65 = 'field1'
    var_66 = 'field2'
    var_67 = {var_65: var_63, var_66: var_64}
    var_68 = module_0.Schema(var_67)
    var_69 = 'field1'
    var_70 = 'value1'
    var_71 = {var_69: var_70}
    var_72 = var_68.validate(var_71)



# Parsed testcases at query #48
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test_schema'
    var_5 = True
    var_6 = module_0.Reference(var_4, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = module_0.Reference(var_4, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Definitions()
    var_13 = module_1.Field()
    var_14 = {var_10: var_13}
    var_15 = module_0.Schema(var_14)
    var_16 = 'user'
    var_17 = module_0.Reference(var_16, var_12)
    var_18 = 'John'
    var_19 = {var_10: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = module_0.Definitions()
    var_22 = module_1.Field()
    var_23 = 'age'
    var_24 = {var_23: var_22}
    var_25 = module_0.Schema(var_24)
    var_26 = 'person'
    var_27 = module_0.Reference(var_26, var_21)
    var_28 = 25
    var_29 = {var_23: var_28}
    var_30 = var_27.validate(var_29)
    var_31 = module_0.Definitions()
    var_32 = 'id'
    var_33 = module_1.Field()
    var_34 = {var_32: var_33}
    var_35 = module_0.Schema(var_34)
    var_36 = module_1.Field()
    var_37 = {var_10: var_36}
    var_38 = module_0.Schema(var_37)
    var_39 = 'schema_a'
    var_40 = module_0.Reference(var_39, var_31)
    var_41 = 'schema_b'
    var_42 = module_0.Reference(var_41, var_31)
    var_43 = {var_32: var_5}
    var_44 = var_40.validate(var_43)
    var_45 = 'test'
    var_46 = {var_10: var_45}
    var_47 = var_42.validate(var_46)
    var_48 = False
    var_49 = module_0.Reference(var_4, var_0)
    var_50 = None
    var_51 = var_49.validate(var_50)



# Parsed testcases at query #49
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'TestSchema'
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Reference(var_5, var_0)
    var_15 = 'John'
    var_16 = {var_12: var_15}
    var_17 = var_14.validate(var_16)
    var_18 = module_0.Definitions()
    var_19 = module_1.Field(allow_null=var_10)
    var_20 = 'required_field'
    var_21 = {var_20: var_19}
    var_22 = module_0.Schema(var_21)
    var_23 = 'StrictSchema'
    var_24 = module_0.Reference(var_23, var_18)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Definitions()
    var_28 = 'id'
    var_29 = 'value'
    var_30 = module_1.Field()
    var_31 = module_1.Field()
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = module_0.Schema(var_32)
    var_34 = 'NestedSchema'
    var_35 = module_0.Reference(var_34, var_27)
    var_36 = 'test'
    var_37 = {var_28: var_6, var_29: var_36}
    var_38 = var_35.validate(var_37)



# Parsed testcases at query #50
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = {var_0: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.String()
    var_13 = {var_10: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = 'invalid'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = {var_15: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = 'invalid_key'
    var_25 = module_0.String()
    var_26 = {var_20: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = 'required'
    var_31 = 'age'
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = {var_28: var_32, var_31: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = 'John'
    var_37 = 30
    var_38 = {var_28: var_36, var_31: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = 'default_name'
    var_41 = module_0.String()
    var_42 = {var_28: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)
    var_46 = module_0.String()
    var_47 = 'id'
    var_48 = module_0.String()
    var_49 = {var_47: var_46, var_28: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = '123'
    var_52 = {var_47: var_51, var_28: var_36}
    var_53 = var_50.validate(var_52)
    var_54 = module_0.String()
    var_55 = {var_28: var_54}
    var_56 = module_1.Schema(var_55)
    var_57 = 'extra'
    var_58 = 'field'
    var_59 = {var_28: var_36, var_57: var_58}
    var_60 = var_56.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = {var_31: var_61}
    var_63 = module_1.Schema(var_62)
    var_64 = 'age'
    var_65 = 'not_an_int'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_64: var_68}
    var_70 = module_1.Schema(var_69)
    var_71 = (var_64, var_36)
    var_72 = [var_71]
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = {var_64: var_73, var_31: var_74}
    var_76 = module_1.Schema(var_75)
    var_77 = 'name'
    var_78 = 'age'
    var_79 = 123
    var_80 = 'invalid'
    var_81 = {var_77: var_79, var_78: var_80}
    var_82 = var_76.validate(var_81)
    var_83 = module_0.String()
    var_84 = 'nickname'
    var_85 = module_0.String()
    var_86 = {var_77: var_85, var_84: var_83}
    var_87 = module_1.Schema(var_86)
    var_88 = {var_77: var_36}
    var_89 = var_87.validate(var_88)



