####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #2
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
    var_10 = 'not_a_dict'
    var_11 = var_5.validate(var_10)
    var_12 = {var_11: var_6}
    var_13 = var_5.validate(var_12)
    var_14 = True
    var_15 = module_1.Schema(var_4)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = False
    var_19 = module_1.Schema(var_4)
    var_20 = var_19.validate(var_16)
    var_21 = 123
    var_22 = {var_21: var_6}
    var_23 = var_5.validate(var_22)
    var_24 = 'Unknown'
    var_25 = module_0.Field(default=var_24)
    var_26 = module_0.Field()
    var_27 = {var_23: var_25, var_1: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = {var_1: var_7}
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Field(read_only=var_14)
    var_32 = module_0.Field()
    var_33 = {var_23: var_31, var_1: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {var_23: var_6, var_1: var_7}
    var_36 = var_34.validate(var_35)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = {var_2: var_1}
    var_8 = False
    var_9 = module_1.Schema(var_7)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = 'not a dict'
    var_13 = var_9.validate(var_12)
    var_14 = 1
    var_15 = 'invalid key'
    var_16 = {var_14: var_15}
    var_17 = var_9.validate(var_16)
    var_18 = module_0.Field()
    var_19 = 'required'
    var_20 = {var_19: var_18}
    var_21 = module_1.Schema(var_20)
    var_22 = {}
    var_23 = var_21.validate(var_22)
    var_24 = 'value'
    var_25 = {var_19: var_24}
    var_26 = var_21.validate(var_25)
    var_27 = module_0.Field(read_only=var_22)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'default'
    var_34 = module_0.Field(default=var_33)
    var_35 = {var_33: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field1'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_1.Field()
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = module_0.Schema(var_6)
    var_8 = 'value'
    var_9 = {var_5: var_8}
    var_10 = var_7.serialize(var_9)
    var_11 = module_1.Field()
    var_12 = {var_5: var_11}
    var_13 = module_0.Schema(var_12)
    var_14 = var_13.serialize(var_9)
    var_15 = module_1.Field()
    var_16 = {var_5: var_15}
    var_17 = module_0.Schema(var_16)
    var_18 = 'other_key'
    var_19 = {var_18: var_8}
    var_20 = var_17.serialize(var_19)
    var_21 = module_1.Field()
    var_22 = {var_5: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = var_23.serialize(var_19)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test_field'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'valid_value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'valid_value'
    var_5 = False
    var_6 = module_0.Reference(var_1, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_7, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None



# Parsed testcases at query #7
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'invalid key'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = module_0.Field()
    var_18 = {var_13: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = 'age'
    var_23 = 18
    var_24 = lambda x: x >= var_23
    var_25 = [var_24]
    var_26 = module_0.Field()
    var_27 = {var_22: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'age'
    var_30 = 16
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Field()
    var_34 = {var_29: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = 'test'
    var_37 = {var_29: var_36}
    var_38 = var_35.validate(var_37)
    var_39 = 'default'
    var_40 = module_0.Field(default=var_39)
    var_41 = {var_29: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 123
    var_4 = var_2.validate(var_3)
    assert var_4 == 123



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
    var_5 = 'test'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'John'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = None
    var_13 = var_6.validate(var_12)
    assert var_13 is None
    var_14 = 123
    var_15 = {var_10: var_14}
    var_16 = var_6.validate(var_15)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'target'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'name'
    var_4 = module_1.Field()
    var_5 = {var_3: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'John'
    var_8 = {var_3: var_7}
    var_9 = var_2.validate(var_8)
    var_10 = None
    var_11 = var_2.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_2.validate(var_12)
    var_14 = 'name'
    var_15 = 123
    var_16 = {var_14: var_15}
    var_17 = var_2.validate(var_16)
    var_18 = 'not an object'
    var_19 = var_2.validate(var_18)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = None
    var_6 = var_2.validate(var_5)
    var_7 = 'valid_value'
    var_8 = var_2.validate(var_7)
    assert var_8 == 'valid_value'
    var_9 = 'invalid_value'
    var_10 = var_2.validate(var_9)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    var_7 = False
    var_8 = module_0.Reference(var_5, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = True
    var_12 = module_0.Reference(var_9, var_0)
    var_13 = var_12.validate(var_10)
    assert var_13 is None



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = None
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'TestSchema'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 'John'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 123
    var_14 = {var_11: var_13}
    var_15 = var_5.validate(var_14)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)
    var_4 = None
    var_5 = var_1.validate(var_4)
    var_6 = 'not a dict'
    var_7 = var_1.validate(var_6)
    var_8 = 'invalid_key'
    var_9 = 1
    var_10 = {var_8: var_9}
    var_11 = var_1.validate(var_10)
    var_12 = 'required_key'
    var_13 = module_1.Field()
    var_14 = {var_12: var_13}
    var_15 = module_0.Schema(var_14)
    var_16 = {}
    var_17 = var_15.validate(var_16)
    var_18 = 'key'
    var_19 = module_1.Field()
    var_20 = {var_18: var_19}
    var_21 = module_0.Schema(var_20)
    var_22 = 'value'
    var_23 = {var_18: var_22}
    var_24 = var_21.validate(var_23)
    var_25 = True
    var_26 = module_1.Field(read_only=var_25)
    var_27 = {var_18: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = {var_18: var_22}
    var_30 = var_28.validate(var_29)
    var_31 = 'default_value'
    var_32 = module_1.Field(default=var_31)
    var_33 = {var_18: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 'nested_key'
    var_38 = module_1.Field()
    var_39 = {var_37: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = 'nested_schema'
    var_42 = {var_41: var_40}
    var_43 = module_0.Schema(var_42)
    var_44 = {var_37: var_22}
    var_45 = {var_41: var_44}
    var_46 = var_43.validate(var_45)
    var_47 = 'nested_schema'
    var_48 = 'nested_key'
    var_49 = 1
    var_50 = {var_48: var_49}
    var_51 = {var_47: var_50}
    var_52 = var_43.validate(var_51)



# Parsed testcases at query #17
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
    var_10 = 'invalid'
    var_11 = var_5.validate(var_10)
    var_12 = None
    var_13 = var_5.validate(var_12)
    var_14 = 'name'
    var_15 = 'John'
    var_16 = {var_14: var_15}
    var_17 = var_5.validate(var_16)
    var_18 = 123
    var_19 = 'age'
    var_20 = 'John'
    var_21 = 30
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = var_5.validate(var_22)
    var_24 = None
    var_25 = var_5.validate(var_24)
    assert var_25 is None
    var_26 = module_0.Field()
    var_27 = 25
    var_28 = module_0.Field(default=var_27)
    var_29 = {var_18: var_26, var_19: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = {var_18: var_22}
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Field()
    var_34 = True
    var_35 = module_0.Field(read_only=var_34)
    var_36 = {var_18: var_33, var_19: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = {var_18: var_22, var_19: var_23}
    var_39 = var_37.validate(var_38)



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    var_5 = 'Person'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'John'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    var_12 = 'name'
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #19
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
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field'
    var_27 = module_1.Field()
    var_28 = {var_26: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = 'field'
    var_31 = 'too long'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_1.Field()
    var_35 = {var_26: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'value'
    var_38 = {var_26: var_37}
    var_39 = var_36.validate(var_38)
    var_40 = 'default'
    var_41 = module_1.Field(default=var_40)
    var_42 = {var_26: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)
    var_46 = module_1.Field(read_only=var_31)
    var_47 = {var_26: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = {var_26: var_37}
    var_50 = var_48.validate(var_49)



# Parsed testcases at query #20
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'invalid key'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = {}
    var_18 = var_8.validate(var_17)
    var_19 = 'test'
    var_20 = {var_17: var_19}
    var_21 = var_8.validate(var_20)
    var_22 = module_0.Field(read_only=var_15)
    var_23 = {var_17: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'default'
    var_28 = module_0.Field(default=var_27)
    var_29 = {var_17: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = module_1.object()
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'foo'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_3)
    var_5 = module_0.Reference(var_2, var_0)
    var_6 = 'bar'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    var_11 = 'foo'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_5.validate(var_13)
    var_15 = 'foo'
    var_16 = None
    var_17 = {var_15: var_16}
    var_18 = var_5.validate(var_17)
    var_19 = 'foo'
    var_20 = []
    var_21 = {var_19: var_20}
    var_22 = var_5.validate(var_21)
    var_23 = 'foo'
    var_24 = {}
    var_25 = {var_23: var_24}
    var_26 = var_5.validate(var_25)
    var_27 = 'foo'
    var_28 = ''
    var_29 = {var_27: var_28}
    var_30 = var_5.validate(var_29)
    var_31 = 'foo'
    var_32 = ''
    var_33 = {var_31: var_32}
    var_34 = var_5.validate(var_33)
    var_35 = 'foo'
    var_36 = ''
    var_37 = {var_35: var_36}
    var_38 = var_5.validate(var_37)
    var_39 = 'foo'
    var_40 = ''
    var_41 = {var_39: var_40}
    var_42 = var_5.validate(var_41)
    var_43 = 'foo'
    var_44 = ''
    var_45 = {var_43: var_44}
    var_46 = var_5.validate(var_45)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'valid'
    var_7 = var_3.validate(var_6)
    var_8 = var_1.validate(var_6)
    var_9 = 'invalid'
    var_10 = var_3.validate(var_9)
    var_11 = var_1.validate(var_9)



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
    var_10 = None
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = var_5.validate(var_11)
    var_13 = {var_0: var_6}
    var_14 = var_5.validate(var_13)
    var_15 = 'thirty'
    var_16 = {var_14: var_6, var_1: var_15}
    var_17 = var_5.validate(var_16)
    var_18 = 1
    var_19 = {var_18: var_6, var_1: var_7}
    var_20 = var_5.validate(var_19)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = {var_2: var_6}
    var_11 = var_5.validate(var_10)
    var_12 = None
    var_13 = var_5.validate(var_12)
    var_14 = 'not a dict'
    var_15 = var_5.validate(var_14)
    var_16 = 'All tests passed for Schema.validate'
    var_17 = print(var_16)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'TestSchema'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 'John'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 123
    var_14 = {var_11: var_13}
    var_15 = var_5.validate(var_14)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = var_2.validate(var_1)
    assert var_3 == 'test'



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'value'
    var_6 = var_2.validate(var_5)
    assert var_6 == 'value'



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_1.Schema(var_5)
    var_7 = 'value1'
    var_8 = None
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = None
    var_14 = {var_11: var_13, var_12: var_13}
    var_15 = var_6.validate(var_14)
    var_16 = 'field1'
    var_17 = 'value1'
    var_18 = {var_16: var_17}
    var_19 = var_6.validate(var_18)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = 'field3'
    var_23 = 'value1'
    var_24 = 'value2'
    var_25 = 'value3'
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = var_6.validate(var_26)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = 1
    var_7 = {var_6: var_2, var_1: var_3}
    var_8 = None
    var_9 = 'not a dict'



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'John'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = module_0.Field()
    var_8 = {var_0: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = {}
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Field()
    var_13 = {var_11: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = 1
    var_16 = {var_15: var_4}
    var_17 = var_14.validate(var_16)
    var_18 = module_0.Field()
    var_19 = {var_17: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = module_0.Field()
    var_24 = {var_22: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = 'John'
    var_27 = var_25.validate(var_26)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = None
    var_7 = True
    var_8 = 'not a dict'
    var_9 = 123
    var_10 = {var_9: var_1}
    var_11 = {}
    var_12 = True
    var_13 = 'value'
    var_14 = {var_0: var_13}
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = {var_15: var_1, var_16: var_1}
    var_18 = {var_15: var_1, var_16: var_3}
    var_19 = {var_15: var_1}



# Parsed testcases at query #33
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'invalid key'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = {}
    var_18 = var_8.validate(var_17)
    var_19 = module_0.Field()
    var_20 = {var_17: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'test'
    var_23 = {var_17: var_22}
    var_24 = var_21.validate(var_23)
    var_25 = 'default'
    var_26 = module_0.Field(default=var_25)
    var_27 = {var_17: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Field(read_only=var_15)
    var_32 = {var_17: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Field()
    var_37 = 'nested'
    var_38 = {var_37: var_36}
    var_39 = module_1.Schema(var_38)
    var_40 = 'nested'
    var_41 = None
    var_42 = {var_40: var_41}
    var_43 = var_39.validate(var_42)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'bar'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_3.validate(var_11)
    var_13 = 123
    var_14 = {var_13: var_4}
    var_15 = var_3.validate(var_14)
    var_16 = {}
    var_17 = var_3.validate(var_16)
    var_18 = 2
    var_19 = module_0.Field()
    var_20 = {var_17: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = {var_17: var_4}
    var_23 = var_21.validate(var_22)
    var_24 = 'default'
    var_25 = module_0.Field(default=var_24)
    var_26 = {var_23: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = True
    var_31 = module_0.Field(read_only=var_30)
    var_32 = {var_23: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {var_23: var_4}
    var_35 = var_33.validate(var_34)



# Parsed testcases at query #35
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
    var_10 = {var_0: var_6}
    var_11 = var_5.validate(var_10)
    var_12 = 1
    var_13 = {var_12: var_6, var_1: var_7}
    var_14 = var_5.validate(var_13)
    var_15 = None
    var_16 = var_5.validate(var_15)
    var_17 = var_5.validate(var_15)
    assert var_17 is None



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = True
    var_3 = module_0.Field()
    var_4 = False
    var_5 = module_0.Field(allow_null=var_2)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = 'age'
    var_13 = 30
    var_14 = {var_12: var_13}
    var_15 = var_7.validate(var_14)
    var_16 = 1
    var_17 = 'age'
    var_18 = 'John'
    var_19 = 30
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = var_7.validate(var_20)
    var_22 = 'name'
    var_23 = 'age'
    var_24 = None
    var_25 = 30
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = var_7.validate(var_26)
    var_28 = None
    var_29 = {var_22: var_8, var_23: var_28}
    var_30 = var_7.validate(var_29)
    var_31 = 'not a dict'
    var_32 = var_7.validate(var_31)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.serialize(var_5)
    var_7 = None
    var_8 = var_3.serialize(var_7)
    assert var_8 is None



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    assert var_4 == 5
    var_5 = None
    var_6 = var_2.validate(var_5)
    assert var_6 is None
    var_7 = None
    var_8 = var_2.validate(var_7)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = var_2.validate(var_1)
    assert var_5 == 'test'
    var_6 = 1
    var_7 = var_2.validate(var_6)
    assert var_7 == 1
    var_8 = True
    var_9 = var_2.validate(var_8)
    assert var_9 is True
    var_10 = False
    var_11 = var_2.validate(var_10)
    assert var_11 is False
    var_12 = []
    var_13 = var_2.validate(var_12)
    var_14 = {}
    var_15 = var_2.validate(var_14)
    var_16 = 'a'
    var_17 = {var_16: var_8}
    var_18 = var_2.validate(var_17)
    var_19 = 'b'
    var_20 = 2
    var_21 = {var_16: var_8, var_19: var_20}
    var_22 = var_2.validate(var_21)



# Parsed testcases at query #9
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = module_0.Field()
    var_18 = {var_13: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = 5
    var_23 = module_0.Field()
    var_24 = {var_20: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = 'name'
    var_27 = 'too long'
    var_28 = {var_26: var_27}
    var_29 = var_25.validate(var_28)
    var_30 = module_0.Field()
    var_31 = {var_26: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = 'valid'
    var_34 = {var_26: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = 'default'
    var_37 = module_0.Field(default=var_36)
    var_38 = {var_26: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field(read_only=var_28)
    var_43 = {var_26: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {}
    var_46 = var_44.validate(var_45)
    var_47 = 'age'
    var_48 = module_0.Field()
    var_49 = 18
    var_50 = module_0.Field()
    var_51 = {var_26: var_48, var_47: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'age'
    var_54 = 15
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)



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
    var_4 = module_1.Field()
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = module_0.Schema(var_6)
    var_8 = 'value'
    var_9 = {var_5: var_8}
    var_10 = var_7.serialize(var_9)
    var_11 = module_1.Field()
    var_12 = {var_5: var_11}
    var_13 = module_0.Schema(var_12)
    var_14 = var_13.serialize(var_9)
    var_15 = module_1.Field()
    var_16 = {var_5: var_15}
    var_17 = module_0.Schema(var_16)
    var_18 = 'other_key'
    var_19 = {var_18: var_8}
    var_20 = var_17.serialize(var_19)
    var_21 = module_1.Field()
    var_22 = {var_5: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = var_23.serialize(var_19)



# Parsed testcases at query #11
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
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = None
    var_13 = var_5.validate(var_12)
    assert var_13 is None
    var_14 = 1
    var_15 = 'age'
    var_16 = 'John'
    var_17 = 30
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = var_5.validate(var_18)
    var_20 = 'name'
    var_21 = 'John'
    var_22 = {var_20: var_21}
    var_23 = var_5.validate(var_22)
    var_24 = module_0.Field()
    var_25 = 25
    var_26 = module_0.Field(default=var_25)
    var_27 = {var_20: var_24, var_21: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = {var_20: var_18}
    var_30 = var_28.validate(var_29)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 'invalid'
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'invalid'
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'value'
    var_5 = None
    var_6 = var_2.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.Reference(var_1, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = True
    var_3 = module_0.Field()
    var_4 = False
    var_5 = module_0.Field()
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = 'age'
    var_13 = 30
    var_14 = {var_12: var_13}
    var_15 = var_7.validate(var_14)
    var_16 = None
    var_17 = var_7.validate(var_16)
    var_18 = 1
    var_19 = 'John'
    var_20 = {var_18: var_19}
    var_21 = var_7.validate(var_20)
    var_22 = 'id'
    var_23 = module_0.Field()
    var_24 = module_0.Field(read_only=var_20)
    var_25 = {var_18: var_23, var_22: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 123
    var_28 = {var_18: var_8, var_22: var_27}
    var_29 = var_26.validate(var_28)
    var_30 = module_0.Field()
    var_31 = 25
    var_32 = module_0.Field(default=var_31)
    var_33 = {var_18: var_30, var_19: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {var_18: var_8}
    var_36 = var_34.validate(var_35)
    var_37 = 'address'
    var_38 = module_0.Field()
    var_39 = {var_37: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = 'details'
    var_42 = module_0.Field()
    var_43 = {var_18: var_42, var_41: var_40}
    var_44 = module_1.Schema(var_43)
    var_45 = '123 Main St'
    var_46 = {var_37: var_45}
    var_47 = {var_18: var_8, var_41: var_46}
    var_48 = var_44.validate(var_47)
    var_49 = module_0.Field()
    var_50 = module_0.Field(allow_null=var_20)
    var_51 = {var_18: var_49, var_19: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = None
    var_54 = {var_18: var_8, var_19: var_53}
    var_55 = var_52.validate(var_54)



# Parsed testcases at query #17
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = module_0.Field()
    var_18 = {var_13: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'test'
    var_26 = {var_20: var_25}
    var_27 = var_24.validate(var_26)
    var_28 = 'default'
    var_29 = module_0.Field(default=var_28)
    var_30 = {var_20: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = {}
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Field(read_only=var_15)
    var_35 = {var_20: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {var_20: var_25}
    var_38 = var_36.validate(var_37)
    var_39 = 'nested'
    var_40 = 'inner'
    var_41 = module_0.Field()
    var_42 = {var_40: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {var_39: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'nested'
    var_47 = {}
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = var_2.validate(var_1)
    assert var_3 == 'test'
    var_4 = None
    var_5 = var_2.validate(var_4)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 'John'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    var_11 = 'name'
    var_12 = 123
    var_13 = {var_11: var_12}
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1
    var_7 = var_2.validate(var_1)
    assert var_7 == 'test'
    var_8 = True
    var_9 = var_2.validate(var_8)
    assert var_9 is True
    var_10 = False
    var_11 = var_2.validate(var_10)
    assert var_11 is False
    var_12 = []
    var_13 = var_2.validate(var_12)
    var_14 = {}
    var_15 = var_2.validate(var_14)
    var_16 = {var_1: var_8}
    var_17 = var_2.validate(var_16)
    var_18 = {var_1: var_1}
    var_19 = var_2.validate(var_18)
    var_20 = True
    var_21 = {var_1: var_20}
    var_22 = var_2.validate(var_21)
    var_23 = {var_1: var_10}
    var_24 = var_2.validate(var_23)
    var_25 = []
    var_26 = {var_1: var_25}
    var_27 = var_2.validate(var_26)
    var_28 = {}
    var_29 = {var_1: var_28}
    var_30 = var_2.validate(var_29)
    var_31 = {var_1: var_20}
    var_32 = {var_1: var_31}
    var_33 = var_2.validate(var_32)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1
    var_7 = var_2.validate(var_1)
    assert var_7 == 'test'
    var_8 = True
    var_9 = var_2.validate(var_8)
    assert var_9 is True
    var_10 = False
    var_11 = var_2.validate(var_10)
    assert var_11 is False
    var_12 = []
    var_13 = var_2.validate(var_12)
    var_14 = {}
    var_15 = var_2.validate(var_14)
    var_16 = {var_1: var_8}
    var_17 = var_2.validate(var_16)
    var_18 = {var_1: var_1}
    var_19 = var_2.validate(var_18)
    var_20 = True
    var_21 = {var_1: var_20}
    var_22 = var_2.validate(var_21)
    var_23 = {var_1: var_10}
    var_24 = var_2.validate(var_23)
    var_25 = []
    var_26 = {var_1: var_25}
    var_27 = var_2.validate(var_26)
    var_28 = {}
    var_29 = {var_1: var_28}
    var_30 = var_2.validate(var_29)
    var_31 = {var_1: var_20}
    var_32 = {var_1: var_31}
    var_33 = var_2.validate(var_32)
    var_34 = {var_1: var_1}
    var_35 = {var_1: var_34}
    var_36 = var_2.validate(var_35)
    var_37 = True
    var_38 = {var_1: var_37}
    var_39 = {var_1: var_38}
    var_40 = var_2.validate(var_39)
    var_41 = {var_1: var_10}
    var_42 = {var_1: var_41}
    var_43 = var_2.validate(var_42)
    var_44 = []
    var_45 = {var_1: var_44}
    var_46 = {var_1: var_45}
    var_47 = var_2.validate(var_46)
    var_48 = {}
    var_49 = {var_1: var_48}
    var_50 = {var_1: var_49}
    var_51 = var_2.validate(var_50)
    var_52 = {var_1: var_37}
    var_53 = {var_1: var_52}
    var_54 = {var_1: var_53}
    var_55 = var_2.validate(var_54)
    var_56 = {var_1: var_1}
    var_57 = {var_1: var_56}
    var_58 = {var_1: var_57}
    var_59 = var_2.validate(var_58)
    var_60 = True
    var_61 = {var_1: var_60}
    var_62 = {var_1: var_61}
    var_63 = {var_1: var_62}
    var_64 = var_2.validate(var_63)
    var_65 = {var_1: var_10}
    var_66 = {var_1: var_65}
    var_67 = {var_1: var_66}
    var_68 = var_2.validate(var_67)
    var_69 = []
    var_70 = {var_1: var_69}
    var_71 = {var_1: var_70}
    var_72 = {var_1: var_71}
    var_73 = var_2.validate(var_72)
    var_74 = {}
    var_75 = {var_1: var_74}
    var_76 = {var_1: var_75}
    var_77 = {var_1: var_76}
    var_78 = var_2.validate(var_77)
    var_79 = {var_1: var_60}
    var_80 = {var_1: var_79}
    var_81 = {var_1: var_80}
    var_82 = {var_1: var_81}
    var_83 = var_2.validate(var_82)
    var_84 = {var_1: var_1}
    var_85 = {var_1: var_84}
    var_86 = {var_1: var_85}
    var_87 = {var_1: var_86}
    var_88 = var_2.validate(var_87)
    var_89 = True
    var_90 = {var_1: var_89}
    var_91 = {var_1: var_90}
    var_92 = {var_1: var_91}
    var_93 = {var_1: var_92}
    var_94 = var_2.validate(var_93)
    var_95 = {var_1: var_10}
    var_96 = {var_1: var_95}
    var_97 = {var_1: var_96}
    var_98 = {var_1: var_97}
    var_99 = var_2.validate(var_98)
    var_100 = []
    var_101 = {var_1: var_100}
    var_102 = {var_1: var_101}
    var_103 = {var_1: var_102}
    var_104 = {var_1: var_103}
    var_105 = var_2.validate(var_104)
    var_106 = {}
    var_107 = {var_1: var_106}
    var_108 = {var_1: var_107}
    var_109 = {var_1: var_108}
    var_110 = {var_1: var_109}
    var_111 = var_2.validate(var_110)



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 'test'
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #23
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
    var_11 = module_0.Field()
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = module_1.Schema(var_12)
    var_14 = {var_0: var_6}
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = {var_15: var_16, var_1: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 'Invalid'
    var_21 = {var_15: var_6, var_7: var_20}
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Field()
    var_24 = module_0.Field()
    var_25 = {var_22: var_23, var_1: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'Not a dictionary'
    var_28 = var_26.validate(var_27)
    var_29 = module_0.Field()
    var_30 = module_0.Field()
    var_31 = {var_28: var_29, var_1: var_30}
    var_32 = True
    var_33 = module_1.Schema(var_31)
    var_34 = None
    var_35 = var_33.validate(var_34)
    assert var_35 is None
    var_36 = module_0.Field()
    var_37 = module_0.Field()
    var_38 = {var_28: var_36, var_1: var_37}
    var_39 = False
    var_40 = module_1.Schema(var_38)
    var_41 = None
    var_42 = var_40.validate(var_41)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'John'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = True
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_1.Schema(var_2)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 123
    var_16 = var_12.validate(var_15)
    var_17 = 123
    var_18 = 'John'
    var_19 = {var_17: var_18}
    var_20 = var_12.validate(var_19)
    var_21 = {}
    var_22 = var_12.validate(var_21)
    var_23 = 'age'
    var_24 = 18
    var_25 = module_0.Field()
    var_26 = {var_23: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = 'age'
    var_29 = 15
    var_30 = {var_28: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = 'address'
    var_33 = 'city'
    var_34 = module_0.Field()
    var_35 = {var_33: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {var_32: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'New York'
    var_40 = {var_33: var_39}
    var_41 = {var_32: var_40}
    var_42 = var_38.validate(var_41)
    var_43 = 'address'
    var_44 = 'city'
    var_45 = 123
    var_46 = {var_44: var_45}
    var_47 = {var_43: var_46}
    var_48 = var_38.validate(var_47)
    var_49 = 'Unknown'
    var_50 = module_0.Field(default=var_49)
    var_51 = {var_43: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {}
    var_54 = var_52.validate(var_53)



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = 'valid'
    var_9 = var_3.validate(var_8)
    assert var_9 == 'valid'
    var_10 = 'invalid'
    var_11 = module_2.Message(text=var_10, code=var_10)
    var_12 = [var_11]
    var_13 = module_2.ValidationError(messages=var_12)
    var_14 = 'invalid'
    var_15 = var_3.validate(var_14)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1
    var_7 = var_2.validate(var_1)
    assert var_7 == 'test'
    var_8 = True
    var_9 = var_2.validate(var_8)
    assert var_9 is True
    var_10 = False
    var_11 = var_2.validate(var_10)
    assert var_11 is False
    var_12 = []
    var_13 = var_2.validate(var_12)
    var_14 = {}
    var_15 = var_2.validate(var_14)
    var_16 = {var_1: var_8}
    var_17 = var_2.validate(var_16)
    var_18 = {var_1: var_1}
    var_19 = var_2.validate(var_18)
    var_20 = True
    var_21 = {var_1: var_20}
    var_22 = var_2.validate(var_21)
    var_23 = {var_1: var_10}
    var_24 = var_2.validate(var_23)
    var_25 = []
    var_26 = {var_1: var_25}
    var_27 = var_2.validate(var_26)
    var_28 = {}
    var_29 = {var_1: var_28}
    var_30 = var_2.validate(var_29)
    var_31 = {var_1: var_20}
    var_32 = {var_1: var_31}
    var_33 = var_2.validate(var_32)
    var_34 = {var_1: var_1}
    var_35 = {var_1: var_34}
    var_36 = var_2.validate(var_35)
    var_37 = True
    var_38 = {var_1: var_37}
    var_39 = {var_1: var_38}
    var_40 = var_2.validate(var_39)
    var_41 = {var_1: var_10}
    var_42 = {var_1: var_41}
    var_43 = var_2.validate(var_42)
    var_44 = []
    var_45 = {var_1: var_44}
    var_46 = {var_1: var_45}
    var_47 = var_2.validate(var_46)
    var_48 = {}
    var_49 = {var_1: var_48}
    var_50 = {var_1: var_49}
    var_51 = var_2.validate(var_50)
    var_52 = {var_1: var_37}
    var_53 = {var_1: var_52}
    var_54 = {var_1: var_53}
    var_55 = var_2.validate(var_54)
    var_56 = {var_1: var_1}
    var_57 = {var_1: var_56}
    var_58 = {var_1: var_57}
    var_59 = var_2.validate(var_58)
    var_60 = True
    var_61 = {var_1: var_60}
    var_62 = {var_1: var_61}
    var_63 = {var_1: var_62}
    var_64 = var_2.validate(var_63)
    var_65 = {var_1: var_10}
    var_66 = {var_1: var_65}
    var_67 = {var_1: var_66}
    var_68 = var_2.validate(var_67)
    var_69 = []
    var_70 = {var_1: var_69}
    var_71 = {var_1: var_70}
    var_72 = {var_1: var_71}
    var_73 = var_2.validate(var_72)
    var_74 = {}
    var_75 = {var_1: var_74}
    var_76 = {var_1: var_75}
    var_77 = {var_1: var_76}
    var_78 = var_2.validate(var_77)
    var_79 = {var_1: var_60}
    var_80 = {var_1: var_79}
    var_81 = {var_1: var_80}
    var_82 = {var_1: var_81}
    var_83 = var_2.validate(var_82)
    var_84 = {var_1: var_1}
    var_85 = {var_1: var_84}
    var_86 = {var_1: var_85}
    var_87 = {var_1: var_86}
    var_88 = var_2.validate(var_87)
    var_89 = True
    var_90 = {var_1: var_89}
    var_91 = {var_1: var_90}
    var_92 = {var_1: var_91}
    var_93 = {var_1: var_92}
    var_94 = var_2.validate(var_93)
    var_95 = {var_1: var_10}
    var_96 = {var_1: var_95}
    var_97 = {var_1: var_96}
    var_98 = {var_1: var_97}
    var_99 = var_2.validate(var_98)
    var_100 = []
    var_101 = {var_1: var_100}
    var_102 = {var_1: var_101}
    var_103 = {var_1: var_102}
    var_104 = {var_1: var_103}
    var_105 = var_2.validate(var_104)
    var_106 = {}
    var_107 = {var_1: var_106}
    var_108 = {var_1: var_107}
    var_109 = {var_1: var_108}
    var_110 = {var_1: var_109}
    var_111 = var_2.validate(var_110)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'a'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 1
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    var_11 = 'a'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_5.validate(var_13)
    var_15 = 'b'
    var_16 = 1
    var_17 = {var_15: var_16}
    var_18 = var_5.validate(var_17)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = var_2.validate(var_1)
    assert var_5 == 'test'



# Parsed testcases at query #29
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'string'
    var_3 = module_1.Field()
    var_4 = {var_1: var_3}
    var_5 = module_0.Schema(var_4)
    var_6 = 'schema'
    var_7 = module_0.Reference(var_6, var_0)
    var_8 = 'John'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = None
    var_12 = var_7.validate(var_11)
    var_13 = 'name'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_7.validate(var_15)



# Parsed testcases at query #30
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'John'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = 'Jane'
    var_8 = {var_1: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = 'Doe'
    var_11 = {var_1: var_10}
    var_12 = var_3.validate(var_11)
    var_13 = 'Smith'
    var_14 = {var_1: var_13}
    var_15 = var_3.validate(var_14)
    var_16 = 'Brown'
    var_17 = {var_1: var_16}
    var_18 = var_3.validate(var_17)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'field1'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.Field(allow_null=var_7)
    var_9 = 'field2'
    var_10 = {var_9: var_8}
    var_11 = module_1.Schema(var_10)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Field()
    var_15 = 'field3'
    var_16 = {var_15: var_14}
    var_17 = module_1.Schema(var_16)
    var_18 = 'not a dict'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Field()
    var_21 = 'field4'
    var_22 = {var_21: var_20}
    var_23 = module_1.Schema(var_22)
    var_24 = 1
    var_25 = 'invalid key'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)
    var_28 = module_0.Field()
    var_29 = 'field5'
    var_30 = {var_29: var_28}
    var_31 = module_1.Schema(var_30)
    var_32 = {}
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Field()
    var_35 = 'field6'
    var_36 = {var_35: var_34}
    var_37 = module_1.Schema(var_36)
    var_38 = 'valid'
    var_39 = {var_35: var_38}
    var_40 = var_37.validate(var_39)
    var_41 = module_0.Field(read_only=var_32)
    var_42 = 'field7'
    var_43 = {var_42: var_41}
    var_44 = module_1.Schema(var_43)
    var_45 = 'read_only'
    var_46 = {var_42: var_45}
    var_47 = var_44.validate(var_46)
    var_48 = 'default'
    var_49 = module_0.Field(default=var_48)
    var_50 = 'field8'
    var_51 = {var_50: var_49}
    var_52 = module_1.Schema(var_51)
    var_53 = {}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Field(allow_null=var_7)
    var_56 = 'field9'
    var_57 = {var_56: var_55}
    var_58 = module_1.Schema(var_57)
    var_59 = 'field9'
    var_60 = None
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)



# Parsed testcases at query #32
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = var_2.validate(var_1)
    assert var_3 == 'test'



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = 'field1'
    var_11 = 'value1'
    var_12 = {var_10: var_11}
    var_13 = var_5.validate(var_12)
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = 'field3'
    var_17 = 'value1'
    var_18 = 'value2'
    var_19 = 'value3'
    var_20 = {var_14: var_17, var_15: var_18, var_16: var_19}
    var_21 = var_5.validate(var_20)
    var_22 = 'value'
    var_23 = var_5.validate(var_22)
    var_24 = 123
    var_25 = var_5.validate(var_24)
    var_26 = None
    var_27 = var_5.validate(var_26)
    var_28 = None
    var_29 = var_5.validate(var_28)
    assert var_29 is None
    var_30 = {var_27: var_18}
    var_31 = var_5.validate(var_30)
    var_32 = {var_27: var_18}
    var_33 = var_5.validate(var_32)
    var_34 = {var_27: var_18}
    var_35 = var_5.validate(var_34)
    var_36 = 'field2'
    var_37 = 'value2'
    var_38 = {var_36: var_37}
    var_39 = var_5.validate(var_38)
    var_40 = {var_37: var_18}
    var_41 = var_5.validate(var_40)
    var_42 = {var_36: var_39, var_37: var_18}
    var_43 = var_5.validate(var_42)
    var_44 = {var_37: var_18}
    var_45 = var_5.validate(var_44)
    var_46 = {var_37: var_18}
    var_47 = var_5.validate(var_46)
    var_48 = {var_37: var_18}
    var_49 = var_5.validate(var_48)
    var_50 = 'field2'
    var_51 = 'value2'
    var_52 = {var_50: var_51}
    var_53 = var_5.validate(var_52)
    var_54 = {var_51: var_18}
    var_55 = var_5.validate(var_54)
    var_56 = {var_50: var_53, var_51: var_18}
    var_57 = var_5.validate(var_56)



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
    var_5 = 'test'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = {var_1: var_5}
    var_8 = var_6.validate(var_7)
    var_9 = None
    var_10 = var_6.validate(var_9)
    assert var_10 is None
    var_11 = 'invalid'
    var_12 = var_6.validate(var_11)
    var_13 = 'name'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_6.validate(var_15)



# Parsed testcases at query #35
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = 'test_value'
    var_9 = var_3.validate(var_8)
    assert var_9 == 'test_value'



# Parsed testcases at query #36
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
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = True
    var_13 = module_1.Schema(var_4)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = 'not a dict'
    var_17 = var_13.validate(var_16)
    var_18 = 1
    var_19 = 'age'
    var_20 = 'John'
    var_21 = 30
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = var_13.validate(var_22)
    var_24 = 'name'
    var_25 = 'John'
    var_26 = {var_24: var_25}
    var_27 = var_13.validate(var_26)
    var_28 = module_0.Field(read_only=var_12)
    var_29 = module_0.Field()
    var_30 = {var_24: var_28, var_25: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = {var_24: var_22, var_25: var_23}
    var_33 = var_31.validate(var_32)
    var_34 = 'Anonymous'
    var_35 = module_0.Field(default=var_34)
    var_36 = module_0.Field()
    var_37 = {var_24: var_35, var_25: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {var_25: var_23}
    var_40 = var_38.validate(var_39)



# Parsed testcases at query #37
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
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = var_5.validate(var_12)
    var_14 = 'name'
    var_15 = 'age'
    var_16 = 'John'
    var_17 = None
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = var_5.validate(var_18)
    var_20 = 'name'
    var_21 = 'age'
    var_22 = 'extra'
    var_23 = 'John'
    var_24 = 30
    var_25 = 'value'
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = var_5.validate(var_26)
    var_28 = 'name'
    var_29 = 'age'
    var_30 = 123
    var_31 = 'John'
    var_32 = 30
    var_33 = 'value'
    var_34 = {var_28: var_31, var_29: var_32, var_30: var_33}
    var_35 = var_5.validate(var_34)
    var_36 = None
    var_37 = var_5.validate(var_36)



# Parsed testcases at query #38
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
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 1
    var_14 = 'invalid key'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = {}
    var_18 = var_8.validate(var_17)
    var_19 = 'test'
    var_20 = {var_17: var_19}
    var_21 = var_8.validate(var_20)
    var_22 = 'default'
    var_23 = module_0.Field(default=var_22)
    var_24 = {var_17: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Field(read_only=var_15)
    var_29 = {var_17: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = {var_17: var_19}
    var_32 = var_30.validate(var_31)
    var_33 = 5
    var_34 = module_0.Field()
    var_35 = {var_17: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = 'name'
    var_38 = 'too long'
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)



# Parsed testcases at query #39
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = var_2.validate(var_1)
    assert var_3 == 'test'
    var_4 = None
    var_5 = var_2.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)



# Parsed testcases at query #40
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = False
    var_6 = module_0.Reference(var_1, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = module_0.Reference(var_7, var_0)
    var_10 = 1
    var_11 = var_9.validate(var_10)
    assert var_11 == 1
    var_12 = module_0.Reference(var_7, var_0)
    var_13 = var_12.validate(var_7)
    assert var_13 == 'test'



