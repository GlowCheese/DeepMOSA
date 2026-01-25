####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = var_5.serialize(var_8)
    var_13 = 'Bob'
    var_14 = {var_0: var_13}
    var_15 = var_5.serialize(var_14)
    var_16 = 'address'
    var_17 = 'street'
    var_18 = 'city'
    var_19 = module_0.Field()
    var_20 = module_0.Field()
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = {var_16: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = '123 Main St'
    var_26 = 'New York'
    var_27 = {var_17: var_25, var_18: var_26}
    var_28 = {var_16: var_27}
    var_29 = var_24.serialize(var_28)



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
    var_10 = True
    var_11 = module_1.Schema(var_4)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_1.Schema(var_4)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_15.validate(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_15.validate(var_22)
    var_24 = 'name'
    var_25 = 'John'
    var_26 = {var_24: var_25}
    var_27 = var_15.validate(var_26)
    var_28 = module_0.Field()
    var_29 = 25
    var_30 = module_0.Field(default=var_29)
    var_31 = {var_24: var_28, var_25: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = {var_24: var_6}
    var_34 = var_32.validate(var_33)
    var_35 = 'id'
    var_36 = module_0.Field()
    var_37 = module_0.Field(read_only=var_10)
    var_38 = {var_24: var_36, var_35: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {var_24: var_6}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field()
    var_43 = module_0.Field()
    var_44 = {var_24: var_42, var_25: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'name'
    var_47 = 'age'
    var_48 = 'John'
    var_49 = -5
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = var_45.validate(var_50)



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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'name'
    var_40 = 'invalid'
    var_41 = {var_39: var_40}
    var_42 = var_36.validate(var_41)
    var_43 = {}
    var_44 = var_36.validate(var_43)
    var_45 = module_0.Field(read_only=var_12)
    var_46 = module_0.Field()
    var_47 = {var_39: var_45, var_40: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_40: var_7}
    var_50 = var_48.validate(var_49)



# Parsed testcases at query #4
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
    var_6 = module_0.Field()
    var_7 = True
    var_8 = module_0.Field(read_only=var_7)
    var_9 = {var_0: var_6, var_1: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = module_0.Field()
    var_12 = 10
    var_13 = module_0.Field(default=var_12)
    var_14 = {var_0: var_11, var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = module_1.Schema(var_4)



# Parsed testcases at query #5
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
    var_9 = var_5.serialize(var_8)
    var_10 = 'Jane'
    var_11 = 25
    var_12 = None
    var_13 = var_5.serialize(var_12)
    assert var_13 is None
    var_14 = 'Bob'
    var_15 = {var_0: var_14}
    var_16 = var_5.serialize(var_15)
    var_17 = 'address'
    var_18 = 'street'
    var_19 = 'city'
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_17: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = '123 Main St'
    var_27 = 'New York'
    var_28 = {var_18: var_26, var_19: var_27}
    var_29 = {var_17: var_28}
    var_30 = var_25.serialize(var_29)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Field()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 123
    var_30 = 'invalid key'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Field()
    var_34 = module_0.Field()
    var_35 = {var_29: var_33, var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = 'name'
    var_38 = 'John'
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_0.Field()
    var_42 = 25
    var_43 = module_0.Field(default=var_42)
    var_44 = {var_37: var_41, var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {var_37: var_6}
    var_47 = var_45.validate(var_46)
    var_48 = 'id'
    var_49 = module_0.Field()
    var_50 = module_0.Field(read_only=var_12)
    var_51 = {var_37: var_49, var_48: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_37: var_6}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Field()
    var_56 = 'Invalid'
    var_57 = 'invalid'
    var_58 = module_2.Message(text=var_56, code=var_57)
    var_59 = [var_58]
    var_60 = module_2.ValidationError(messages=var_59)
    var_61 = (var_14, var_60)
    var_62 = {var_37: var_55}
    var_63 = module_1.Schema(var_62)
    var_64 = 'name'
    var_65 = 'invalid value'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)



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
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'address'
    var_13 = 'city'
    var_14 = 'zip'
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = {var_12: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 'New York'
    var_22 = '10001'
    var_23 = {var_13: var_21, var_14: var_22}
    var_24 = {var_12: var_23}
    var_25 = var_20.serialize(var_24)
    var_26 = {var_0: var_6}
    var_27 = var_5.serialize(var_26)
    var_28 = {var_0: var_6}



# Parsed testcases at query #8
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)



# Parsed testcases at query #9
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = {var_0: var_6}
    var_13 = var_5.serialize(var_12)
    var_14 = 'address'
    var_15 = 'city'
    var_16 = 'zip'
    var_17 = module_0.Field()
    var_18 = module_0.Field()
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = {var_14: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = 'NYC'
    var_24 = '10001'
    var_25 = {var_15: var_23, var_16: var_24}
    var_26 = {var_14: var_25}
    var_27 = var_22.serialize(var_26)



# Parsed testcases at query #10
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Jane'
    var_13 = 25
    var_14 = 'Bob'
    var_15 = {var_0: var_14}
    var_16 = var_5.serialize(var_15)
    var_17 = 40
    var_18 = {var_1: var_17}
    var_19 = var_5.serialize(var_18)
    var_20 = 'address'
    var_21 = 'street'
    var_22 = 'city'
    var_23 = module_0.Field()
    var_24 = module_0.Field()
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = {var_20: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = '123 Main St'
    var_30 = 'New York'
    var_31 = {var_21: var_29, var_22: var_30}
    var_32 = {var_20: var_31}
    var_33 = var_28.serialize(var_32)
    var_34 = 'id'
    var_35 = module_0.Field()
    var_36 = True
    var_37 = module_0.Field(read_only=var_36)
    var_38 = {var_0: var_35, var_34: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = 'Alice'
    var_41 = {var_0: var_40, var_34: var_36}
    var_42 = var_39.serialize(var_41)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_field'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #12
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
    var_9 = var_5.serialize(var_8)
    var_10 = 'address'
    var_11 = 'city'
    var_12 = 'street'
    var_13 = module_0.Field()
    var_14 = module_0.Field()
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_10: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'New York'
    var_20 = '123 Main St'
    var_21 = {var_11: var_19, var_12: var_20}
    var_22 = {var_10: var_21}
    var_23 = var_18.serialize(var_22)
    var_24 = None
    var_25 = var_5.serialize(var_24)
    assert var_25 is None
    var_26 = 'Jane'
    var_27 = 25
    var_28 = {var_0: var_6}
    var_29 = var_5.serialize(var_28)
    var_30 = 'extra'
    var_31 = 'value'
    var_32 = {var_0: var_6, var_1: var_7, var_30: var_31}
    var_33 = var_5.serialize(var_32)



# Parsed testcases at query #13
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Bob'
    var_13 = {var_0: var_12}
    var_14 = var_5.serialize(var_13)
    var_15 = 'user'
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = {var_15: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'Alice'
    var_23 = 28
    var_24 = {var_0: var_22, var_1: var_23}
    var_25 = {var_15: var_24}
    var_26 = var_21.serialize(var_25)
    var_27 = 'custom'
    var_28 = 'value'
    var_29 = {var_27: var_28}



# Parsed testcases at query #14
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
    var_9 = var_5.serialize(var_8)
    var_10 = 'user'
    var_11 = module_0.Field()
    var_12 = module_0.Field()
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = {var_10: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_0: var_6, var_1: var_7}
    var_18 = {var_10: var_17}
    var_19 = var_16.serialize(var_18)
    var_20 = None
    var_21 = var_5.serialize(var_20)
    assert var_21 is None
    var_22 = {var_0: var_6}
    var_23 = var_5.serialize(var_22)
    var_24 = 'extra'
    var_25 = 'field'
    var_26 = {var_0: var_6, var_1: var_7, var_24: var_25}
    var_27 = var_5.serialize(var_26)



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_4.validate(var_9)
    var_11 = None
    var_12 = var_4.validate(var_11)



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
    var_10 = module_0.Field()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_1.Schema(var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.Field()
    var_17 = {var_0: var_16}
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = 'street'
    var_57 = 'city'
    var_58 = module_0.Field()
    var_59 = module_0.Field()
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = 'address'
    var_63 = module_0.Field()
    var_64 = {var_38: var_63, var_62: var_61}
    var_65 = module_1.Schema(var_64)
    var_66 = 'name'
    var_67 = 'address'
    var_68 = 'John'
    var_69 = 'street'
    var_70 = '123 Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #18
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'name'
    var_40 = 'invalid'
    var_41 = {var_39: var_40}
    var_42 = var_36.validate(var_41)
    var_43 = {}
    var_44 = var_36.validate(var_43)
    var_45 = module_0.Field(read_only=var_12)
    var_46 = {var_39: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #19
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
    var_9 = var_5.serialize(var_8)
    var_10 = {var_0: var_6}
    var_11 = var_5.serialize(var_10)
    var_12 = None
    var_13 = var_5.serialize(var_12)
    assert var_13 is None
    var_14 = 'address'
    var_15 = 'street'
    var_16 = 'city'
    var_17 = module_0.Field()
    var_18 = module_0.Field()
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = {var_14: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = '123 Main'
    var_24 = 'NYC'
    var_25 = {var_15: var_23, var_16: var_24}
    var_26 = {var_14: var_25}
    var_27 = var_22.serialize(var_26)
    var_28 = 'id'
    var_29 = True
    var_30 = module_0.Field(read_only=var_29)
    var_31 = module_0.Field()
    var_32 = {var_28: var_30, var_0: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'Test'
    var_35 = {var_28: var_29, var_0: var_34}
    var_36 = var_33.serialize(var_35)



# Parsed testcases at query #20
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'extra'
    var_13 = 'field'
    var_14 = {var_0: var_6, var_1: var_7, var_12: var_13}
    var_15 = var_5.serialize(var_14)
    var_16 = {var_0: var_6}
    var_17 = var_5.serialize(var_16)
    var_18 = 'user'
    var_19 = module_0.Field()
    var_20 = module_0.Field()
    var_21 = {var_0: var_19, var_1: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = {var_18: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = {var_0: var_6, var_1: var_7}
    var_26 = {var_18: var_25}
    var_27 = var_24.serialize(var_26)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'any_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #22
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Jane'
    var_13 = 25
    var_14 = 'Bob'
    var_15 = {var_0: var_14}
    var_16 = var_5.serialize(var_15)
    var_17 = 'user'
    var_18 = module_0.Field()
    var_19 = module_0.Field()
    var_20 = {var_0: var_18, var_1: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = {var_17: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Alice'
    var_25 = 20
    var_26 = {var_0: var_24, var_1: var_25}
    var_27 = {var_17: var_26}
    var_28 = var_23.serialize(var_27)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_field'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #24
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
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = 'test_value'
    var_9 = var_3.validate(var_8)
    assert var_9 == 'test_value'
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'test_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_ref'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #26
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_ref'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'default_name'
    var_40 = module_0.Field(default=var_39)
    var_41 = {var_37: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Field(read_only=var_12)
    var_46 = module_0.Field()
    var_47 = {var_37: var_45, var_38: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_38: var_7}
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Field()
    var_52 = 'error'
    var_53 = module_2.Message(text=var_52, code=var_52)
    var_54 = [var_53]
    var_55 = module_2.ValidationError(messages=var_54)
    var_56 = (var_14, var_55)
    var_57 = {var_37: var_51}
    var_58 = module_1.Schema(var_57)
    var_59 = 'name'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Field()
    var_57 = 'Invalid'
    var_58 = 'invalid'
    var_59 = module_2.Message(text=var_57, code=var_58)
    var_60 = [var_59]
    var_61 = module_2.ValidationError(messages=var_60)
    var_62 = (var_14, var_61)
    var_63 = {var_38: var_56}
    var_64 = module_1.Schema(var_63)
    var_65 = 'name'
    var_66 = 'invalid'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #31
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'name'
    var_43 = 'invalid'
    var_44 = {var_42: var_43}
    var_45 = var_37.validate(var_44)
    var_46 = {}
    var_47 = var_37.validate(var_46)
    var_48 = module_0.Field()
    var_49 = {var_43: var_7}
    var_50 = var_37.validate(var_49)



# Parsed testcases at query #32
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 5
    var_43 = module_0.Field()
    var_44 = {var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'name'
    var_47 = 'John'
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = 'Default'
    var_51 = module_0.Field(default=var_50)
    var_52 = {var_46: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Field(read_only=var_12)
    var_57 = module_0.Field()
    var_58 = {var_46: var_56, var_47: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = {var_47: var_7}
    var_61 = var_59.validate(var_60)



# Parsed testcases at query #33
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'test_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'validated_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'another_test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'another_value'



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
    var_5 = 'test_schema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'John'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = True
    var_11 = module_0.Reference(var_5, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Reference(var_5, var_0)
    var_16 = var_15.validate(var_12)
    var_17 = 'not a dict'
    var_18 = var_6.validate(var_17)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Field()
    var_57 = 'Invalid'
    var_58 = 'invalid'
    var_59 = module_2.Message(text=var_57, code=var_58)
    var_60 = [var_59]
    var_61 = module_2.ValidationError(messages=var_60)
    var_62 = (var_14, var_61)
    var_63 = {var_38: var_56}
    var_64 = module_1.Schema(var_63)
    var_65 = 'name'
    var_66 = 'invalid'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_9 = var_5.serialize(var_8)
    var_10 = 'Jane'
    var_11 = 25
    var_12 = None
    var_13 = var_5.serialize(var_12)
    assert var_13 is None
    var_14 = 'Bob'
    var_15 = {var_0: var_14}
    var_16 = var_5.serialize(var_15)
    var_17 = 'city'
    var_18 = 'Alice'
    var_19 = 28
    var_20 = 'NYC'
    var_21 = {var_0: var_18, var_1: var_19, var_17: var_20}
    var_22 = var_5.serialize(var_21)
    var_23 = 'address'
    var_24 = 'street'
    var_25 = module_0.Field()
    var_26 = module_0.Field()
    var_27 = {var_24: var_25, var_17: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = {var_23: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = '123 Main St'
    var_32 = 'Boston'
    var_33 = {var_24: var_31, var_17: var_32}
    var_34 = {var_23: var_33}
    var_35 = var_30.serialize(var_34)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = True
    var_5 = module_0.Field(allow_null=var_4)
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
    var_18 = 'not a dict'
    var_19 = var_7.validate(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_7.validate(var_22)
    var_24 = 'default'
    var_25 = module_0.Field(default=var_24, allow_null=var_4)
    var_26 = {var_20: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Field(read_only=var_4)
    var_31 = {var_20: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = module_1.Field()
    var_3 = True
    var_4 = module_1.Field(read_only=var_3)
    var_5 = 'default_value'
    var_6 = module_1.Field(default=var_5)
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Schema(var_10)
    var_12 = {var_7: var_2}
    var_13 = 'Test schema'
    var_14 = module_0.Schema(var_12)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'test_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'validated_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'another_test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'another_value'



# Parsed testcases at query #5
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
    var_8 = module_0.Field()
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'default_name'
    var_12 = module_0.Field(default=var_11)
    var_13 = module_0.Field()
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = module_1.Schema(var_18)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = module_0.Field(allow_null=var_2)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = True
    var_12 = module_1.Schema(var_5)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = None
    var_16 = var_6.validate(var_15)
    var_17 = 'not a dict'
    var_18 = var_6.validate(var_17)
    var_19 = 123
    var_20 = 'invalid key'
    var_21 = {var_19: var_20}
    var_22 = var_6.validate(var_21)
    var_23 = 'name'
    var_24 = 'John'
    var_25 = {var_23: var_24}
    var_26 = var_6.validate(var_25)
    var_27 = 3
    var_28 = module_0.Field(allow_null=var_25)
    var_29 = {var_23: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'name'
    var_32 = 'Jo'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'Default'
    var_36 = module_0.Field(default=var_35, allow_null=var_33)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field(allow_null=var_33, read_only=var_11)
    var_42 = {var_31: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)



# Parsed testcases at query #8
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
    var_9 = var_5.serialize(var_8)
    var_10 = 'user'
    var_11 = 'status'
    var_12 = module_0.Field()
    var_13 = module_0.Field()
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = module_0.Field()
    var_17 = {var_10: var_15, var_11: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'Alice'
    var_20 = 25
    var_21 = {var_0: var_19, var_1: var_20}
    var_22 = 'active'
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = var_18.serialize(var_23)
    var_25 = 'Bob'
    var_26 = 35
    var_27 = None
    var_28 = var_5.serialize(var_27)
    assert var_28 is None
    var_29 = 'Charlie'
    var_30 = {var_0: var_29}
    var_31 = var_5.serialize(var_30)
    var_32 = 'extra'
    var_33 = 'Dave'
    var_34 = 40
    var_35 = 'ignored'
    var_36 = {var_0: var_33, var_1: var_34, var_32: var_35}
    var_37 = var_5.serialize(var_36)



# Parsed testcases at query #9
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Bob'
    var_13 = {var_0: var_12}
    var_14 = var_5.serialize(var_13)
    var_15 = 'address'
    var_16 = 'street'
    var_17 = 'city'
    var_18 = module_0.Field()
    var_19 = module_0.Field()
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = {var_15: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = '123 Main St'
    var_25 = 'New York'
    var_26 = {var_16: var_24, var_17: var_25}
    var_27 = {var_15: var_26}
    var_28 = var_23.serialize(var_27)
    var_29 = {}
    var_30 = var_5.serialize(var_29)



# Parsed testcases at query #10
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Bob'
    var_13 = {var_0: var_12}
    var_14 = var_5.serialize(var_13)
    var_15 = 'address'
    var_16 = 'street'
    var_17 = 'city'
    var_18 = module_0.Field()
    var_19 = module_0.Field()
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = {var_15: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = '123 Main'
    var_25 = 'Springfield'
    var_26 = {var_16: var_24, var_17: var_25}
    var_27 = {var_15: var_26}
    var_28 = var_23.serialize(var_27)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'any_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #12
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'default_name'
    var_40 = module_0.Field(default=var_39)
    var_41 = {var_37: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Field()
    var_46 = {var_38: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = 'age'
    var_49 = -1
    var_50 = {var_48: var_49}
    var_51 = var_47.validate(var_50)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'default_name'
    var_40 = module_0.Field(default=var_39)
    var_41 = {var_37: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Field(read_only=var_12)
    var_46 = {var_37: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)
    var_50 = module_0.Field()
    var_51 = 'valid'
    var_52 = 'invalid'
    var_53 = module_2.Message(text=var_52, code=var_52)
    var_54 = [var_53]
    var_55 = module_2.ValidationError(messages=var_54)
    var_56 = (var_14, var_55)
    var_57 = 'nested'
    var_58 = {var_57: var_50}
    var_59 = module_1.Schema(var_58)
    var_60 = 'nested'
    var_61 = 'invalid'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test_schema'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 'test'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_4, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = None
    var_14 = var_5.validate(var_13)
    var_15 = 'invalid_key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_5.validate(var_17)
    var_19 = 'name'
    var_20 = None
    var_21 = {var_19: var_20}
    var_22 = var_5.validate(var_21)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_field'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'error_field'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #18
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'John'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = 'street'
    var_57 = 'city'
    var_58 = module_0.Field()
    var_59 = module_0.Field()
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = 'address'
    var_63 = module_0.Field()
    var_64 = {var_38: var_63, var_62: var_61}
    var_65 = module_1.Schema(var_64)
    var_66 = '123 Main'
    var_67 = 'Springfield'
    var_68 = {var_56: var_66, var_57: var_67}
    var_69 = {var_38: var_6, var_62: var_68}
    var_70 = var_65.validate(var_69)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_4.validate(var_9)
    var_11 = None
    var_12 = var_4.validate(var_11)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Field()
    var_57 = 'error'
    var_58 = module_2.Message(text=var_57, code=var_57)
    var_59 = [var_58]
    var_60 = module_2.ValidationError(messages=var_59)
    var_61 = (var_14, var_60)
    var_62 = {var_38: var_56}
    var_63 = module_1.Schema(var_62)
    var_64 = 'name'
    var_65 = 'John'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = True
    var_5 = module_0.Field(allow_null=var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = module_1.Schema(var_6)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = None
    var_16 = var_7.validate(var_15)
    var_17 = 'not a dict'
    var_18 = var_7.validate(var_17)
    var_19 = 123
    var_20 = 'invalid key'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = module_0.Field(allow_null=var_25)
    var_28 = 'Invalid'
    var_29 = 'invalid'
    var_30 = module_2.Message(text=var_28, code=var_29)
    var_31 = [var_30]
    var_32 = module_2.ValidationError(messages=var_31)
    var_33 = (var_13, var_32)
    var_34 = {var_23: var_27}
    var_35 = module_1.Schema(var_34)
    var_36 = 'name'
    var_37 = 'invalid'
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = 'default'
    var_41 = module_0.Field(default=var_40, allow_null=var_38)
    var_42 = {var_36: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = True
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = module_0.Field()
    var_12 = False
    var_13 = module_0.Field()
    var_14 = {var_0: var_11, var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = {var_0: var_7}
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Field()
    var_19 = module_0.Field()
    var_20 = {var_0: var_18, var_1: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = {var_0: var_7}
    var_23 = var_21.validate(var_22)
    var_24 = 'required'
    var_25 = module_0.Field()
    var_26 = {var_23: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = 123
    var_29 = {var_28: var_7}
    var_30 = var_27.validate(var_29)
    var_31 = 'invalid_key'
    var_32 = module_0.Field()
    var_33 = {var_30: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = None
    var_36 = var_34.validate(var_35)
    assert var_36 is None
    var_37 = module_0.Field()
    var_38 = {var_30: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = None
    var_41 = var_39.validate(var_40)
    var_42 = 'null'
    var_43 = module_0.Field()
    var_44 = {var_40: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'not a dict'
    var_47 = var_45.validate(var_46)
    var_48 = 'type'
    var_49 = module_0.Field()
    var_50 = 25
    var_51 = module_0.Field(default=var_50)
    var_52 = {var_46: var_49, var_47: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_46: var_7}
    var_55 = var_53.validate(var_54)
    var_56 = 'id'
    var_57 = module_0.Field()
    var_58 = module_0.Field(read_only=var_2)
    var_59 = {var_46: var_57, var_56: var_58}
    var_60 = module_1.Schema(var_59)
    var_61 = {var_46: var_7, var_56: var_2}
    var_62 = var_60.validate(var_61)
    var_63 = module_0.Field()
    var_64 = lambda x: x > var_12
    var_65 = [var_64]
    var_66 = module_0.Field()
    var_67 = {var_46: var_63, var_47: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = -5
    var_70 = {var_46: var_7, var_47: var_69}
    var_71 = var_68.validate(var_70)
    var_72 = 'invalid'



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 1
    var_11 = 0
    var_12 = var_10 / var_11
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'test_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'validated_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_4.validate(var_9)
    var_11 = module_1.Field()
    var_12 = 0
    var_13 = var_9 / var_12
    var_14 = 'error_field'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'any_value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test_schema'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = 'test'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'invalid'
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'any_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'invalid_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #29
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
    var_18 = False
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Field()
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'not a dict'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field()
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 123
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = module_0.Field()
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'John'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 25
    var_44 = module_0.Field(default=var_43)
    var_45 = {var_38: var_42, var_39: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'id'
    var_50 = module_0.Field()
    var_51 = module_0.Field(read_only=var_12)
    var_52 = {var_38: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = True
    var_5 = module_0.Field(allow_null=var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = module_1.Schema(var_6)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = None
    var_16 = var_7.validate(var_15)
    var_17 = 'not a dict'
    var_18 = var_7.validate(var_17)
    var_19 = 123
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = 3
    var_28 = module_0.Field(allow_null=var_25)
    var_29 = {var_23: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'name'
    var_32 = 'Jo'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'Default'
    var_36 = module_0.Field(default=var_35, allow_null=var_33)
    var_37 = module_0.Field(allow_null=var_4)
    var_38 = {var_31: var_36, var_32: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {var_32: var_9}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field(allow_null=var_33, read_only=var_4)
    var_43 = module_0.Field(allow_null=var_4)
    var_44 = {var_31: var_42, var_32: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {var_32: var_9}
    var_47 = var_45.validate(var_46)



