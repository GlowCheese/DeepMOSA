####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_7.validate(var_18)
    var_20 = 123
    var_21 = 'invalid key'
    var_22 = {var_20: var_21}
    var_23 = var_7.validate(var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_7.validate(var_26)
    var_28 = 'invalid'
    var_29 = lambda x: x != var_28
    var_30 = [var_29]
    var_31 = module_0.Field(allow_null=var_26)
    var_32 = {var_24: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'name'
    var_35 = 'invalid'
    var_36 = {var_34: var_35}
    var_37 = var_33.validate(var_36)
    var_38 = 'default_name'
    var_39 = module_0.Field(default=var_38, allow_null=var_36)
    var_40 = {var_34: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Field(allow_null=var_36, read_only=var_4)
    var_45 = {var_34: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_34: var_8}
    var_48 = var_46.validate(var_47)



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
    var_28 = 'Jane'
    var_29 = 25



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
    var_5 = True
    var_6 = module_1.Schema(var_4)
    var_7 = module_0.Field(read_only=var_5)
    var_8 = module_0.Field()
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'default_name'
    var_12 = module_0.Field(default=var_11)
    var_13 = module_0.Field()
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = {}
    var_17 = module_1.Schema(var_16)



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
    var_31 = 'id'
    var_32 = True
    var_33 = module_0.Field(read_only=var_32)
    var_34 = module_0.Field()
    var_35 = {var_31: var_33, var_0: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = 'Alice'
    var_38 = {var_31: var_32, var_0: var_37}
    var_39 = var_36.serialize(var_38)



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
    var_31 = '456 Oak Ave'
    var_32 = 'Boston'



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_input'
    var_5 = var_3.validate(var_4)
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = 'test_input'
    var_12 = var_3.validate(var_11)
    assert var_12 == 'mocked_value'



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
    var_56 = 'street'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'address'
    var_61 = module_0.Field()
    var_62 = {var_38: var_61, var_60: var_59}
    var_63 = module_1.Schema(var_62)
    var_64 = 'name'
    var_65 = 'address'
    var_66 = 'John'
    var_67 = 'city'
    var_68 = 'NYC'
    var_69 = {var_67: var_68}
    var_70 = {var_64: var_66, var_65: var_69}
    var_71 = var_63.validate(var_70)



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
    var_30 = 1
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



# Parsed testcases at query #9
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_15.validate(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_15.validate(var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_15.validate(var_26)
    var_28 = module_0.Field(allow_null=var_26)
    var_29 = lambda x: x > var_26
    var_30 = [var_29]
    var_31 = module_0.Field(allow_null=var_26)
    var_32 = {var_24: var_28, var_25: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'name'
    var_35 = 'age'
    var_36 = 'John'
    var_37 = -1
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = var_33.validate(var_38)
    var_40 = module_0.Field(allow_null=var_36)
    var_41 = 25
    var_42 = module_0.Field(default=var_41, allow_null=var_38)
    var_43 = {var_34: var_40, var_35: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {var_34: var_8}
    var_46 = var_44.validate(var_45)
    var_47 = 'id'
    var_48 = module_0.Field(allow_null=var_36)
    var_49 = module_0.Field(allow_null=var_36, read_only=var_38)
    var_50 = {var_34: var_48, var_47: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_34: var_8}
    var_53 = var_51.validate(var_52)



# Parsed testcases at query #10
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
    var_30 = 'value'
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
    var_55 = 123
    var_56 = {var_37: var_6, var_48: var_55}
    var_57 = var_52.validate(var_56)
    var_58 = 'city'
    var_59 = module_0.Field()
    var_60 = {var_58: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = 'address'
    var_63 = module_0.Field()
    var_64 = {var_37: var_63, var_62: var_61}
    var_65 = module_1.Schema(var_64)
    var_66 = 'name'
    var_67 = 'address'
    var_68 = 'John'
    var_69 = 'street'
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



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
    var_5 = 'test_schema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'value'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'not a dict'
    var_15 = var_6.validate(var_14)



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
    var_42 = 'age'
    var_43 = -5
    var_44 = {var_42: var_43}
    var_45 = var_37.validate(var_44)
    var_46 = {}
    var_47 = var_37.validate(var_46)
    var_48 = 'id'
    var_49 = module_0.Field(read_only=var_12)
    var_50 = module_0.Field()
    var_51 = {var_48: var_49, var_42: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_42: var_6}
    var_54 = var_52.validate(var_53)
    var_55 = 'user'
    var_56 = module_0.Field()
    var_57 = 'user'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = 'John'
    var_61 = -5
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = {var_57: var_62}
    var_64 = var_52.validate(var_63)



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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
    var_5 = 'test_schema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



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
    var_10 = 'invalid_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_46 = {var_37: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)
    var_50 = module_0.Field()
    var_51 = 'error'
    var_52 = module_2.Message(text=var_51, code=var_51)
    var_53 = [var_52]
    var_54 = module_2.ValidationError(messages=var_53)
    var_55 = (var_14, var_54)
    var_56 = {var_37: var_50}
    var_57 = module_1.Schema(var_56)
    var_58 = 'name'
    var_59 = 'value'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_ref'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    assert var_8 is None
    var_9 = False
    var_10 = module_0.Reference(var_3, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_1.Field()
    var_14 = var_11 / var_9
    var_15 = 'error_ref'
    var_16 = module_0.Reference(var_15, var_0)
    var_17 = 'any_value'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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
    var_12 = 'Unknown'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_1: var_9}
    var_18 = var_16.validate(var_17)
    var_19 = module_1.Schema(var_6)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = module_1.Schema(var_6)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = 'not a dict'
    var_26 = var_7.validate(var_25)
    var_27 = 1
    var_28 = 'value'
    var_29 = {var_27: var_28}
    var_30 = var_7.validate(var_29)
    var_31 = 'age'
    var_32 = 30
    var_33 = {var_31: var_32}
    var_34 = var_7.validate(var_33)
    var_35 = module_0.Field(allow_null=var_33)
    var_36 = lambda x: x > var_33
    var_37 = [var_36]
    var_38 = module_0.Field(allow_null=var_33)
    var_39 = {var_31: var_35, var_32: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = 'name'
    var_42 = 'age'
    var_43 = 'John'
    var_44 = -5
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = var_40.validate(var_45)
    var_47 = 'id'
    var_48 = module_0.Field(allow_null=var_43)
    var_49 = module_0.Field(allow_null=var_43, read_only=var_45)
    var_50 = {var_41: var_48, var_47: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_41: var_8}
    var_53 = var_51.validate(var_52)



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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = 'test'
    var_9 = var_3.validate(var_8)
    var_10 = 'Error'
    var_11 = 'error'
    var_12 = module_2.Message(text=var_10, code=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = var_3.validate(var_8)



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'error'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_ref'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #28
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
    var_49 = 3
    var_50 = module_0.Field()
    var_51 = {var_38: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'Jo'
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)



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



# Parsed testcases at query #30
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'error'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #31
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
    var_57 = 'name'
    var_58 = 'age'
    var_59 = 'John'
    var_60 = -5
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = var_53.validate(var_61)



# Parsed testcases at query #33
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
    var_56 = 123
    var_57 = {var_38: var_6, var_49: var_56}
    var_58 = var_53.validate(var_57)
    var_59 = module_0.Field()
    var_60 = 'Nested error'
    var_61 = 'nested'
    var_62 = module_2.Message(text=var_60, code=var_61)
    var_63 = [var_62]
    var_64 = module_2.ValidationError(messages=var_63)
    var_65 = (var_14, var_64)
    var_66 = {var_61: var_59}
    var_67 = module_1.Schema(var_66)
    var_68 = 'nested'
    var_69 = 'invalid'
    var_70 = {var_68: var_69}
    var_71 = var_67.validate(var_70)



# Parsed testcases at query #34
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
    var_56 = {var_38: var_6, var_49: var_12}
    var_57 = var_53.validate(var_56)
    var_58 = module_0.Field()
    var_59 = 'error'
    var_60 = module_2.Message(text=var_59, code=var_59)
    var_61 = [var_60]
    var_62 = module_2.ValidationError(messages=var_61)
    var_63 = (var_14, var_62)
    var_64 = {var_38: var_58}
    var_65 = module_1.Schema(var_64)
    var_66 = 'name'
    var_67 = 'invalid'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)



# Parsed testcases at query #36
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
    var_5 = 'test_schema'
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #38
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
    var_46 = 'Not a string'
    var_47 = 'invalid'
    var_48 = module_2.Message(text=var_46, code=var_47)
    var_49 = [var_48]
    var_50 = module_2.ValidationError(messages=var_49)
    var_51 = (var_14, var_50)
    var_52 = {var_37: var_45}
    var_53 = module_1.Schema(var_52)
    var_54 = 'john'
    var_55 = {var_37: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = 'name'
    var_58 = 123
    var_59 = {var_57: var_58}
    var_60 = var_53.validate(var_59)



# Parsed testcases at query #39
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'invalid'
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #40
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'not a dict'
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #41
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



# Parsed testcases at query #42
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
    var_12 = {var_0: var_8}
    var_13 = var_7.validate(var_12)
    var_14 = None
    var_15 = {var_0: var_8, var_1: var_14}
    var_16 = var_7.validate(var_15)
    var_17 = 'age'
    var_18 = 30
    var_19 = {var_17: var_18}
    var_20 = var_7.validate(var_19)
    var_21 = 123
    var_22 = 'John'
    var_23 = {var_21: var_22}
    var_24 = var_7.validate(var_23)
    var_25 = 'not a dict'
    var_26 = var_7.validate(var_25)
    var_27 = 'name'
    var_28 = None
    var_29 = {var_27: var_28}
    var_30 = var_7.validate(var_29)
    var_31 = 'DefaultName'
    var_32 = module_0.Field(default=var_31, allow_null=var_29)
    var_33 = module_0.Field(allow_null=var_4)
    var_34 = {var_27: var_32, var_28: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = {var_28: var_9}
    var_37 = var_35.validate(var_36)
    var_38 = module_0.Field(allow_null=var_29)
    var_39 = module_0.Field(allow_null=var_4, read_only=var_4)
    var_40 = {var_27: var_38, var_28: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = {var_27: var_8}
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #43
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
    var_56 = 'value'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'data'
    var_61 = {var_60: var_59}
    var_62 = module_1.Schema(var_61)
    var_63 = 'data'
    var_64 = 'invalid'
    var_65 = 'key'
    var_66 = {var_64: var_65}
    var_67 = {var_63: var_66}
    var_68 = var_62.validate(var_67)



# Parsed testcases at query #44
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'test'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #45
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'test_error'
    var_12 = 'test_code'
    var_13 = module_2.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'invalid_value'
    var_17 = var_3.validate(var_16)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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
    var_66 = 'invalid_value'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #48
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
    var_24 = 'default_name'
    var_25 = module_0.Field(default=var_24, allow_null=var_22)
    var_26 = 25
    var_27 = module_0.Field(default=var_26, allow_null=var_4)
    var_28 = {var_20: var_25, var_21: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = 'id'
    var_33 = module_0.Field(allow_null=var_22)
    var_34 = module_0.Field(allow_null=var_22, read_only=var_4)
    var_35 = {var_20: var_33, var_32: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {var_20: var_8}
    var_38 = var_36.validate(var_37)
    var_39 = 'user'
    var_40 = module_0.Field(allow_null=var_22)
    var_41 = {var_20: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_39: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = 'user'
    var_46 = {}
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_19 = var_5.validate(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_5.validate(var_22)
    var_24 = module_0.Field()
    var_25 = module_0.Field()
    var_26 = {var_20: var_24, var_21: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = 'name'
    var_29 = 'John'
    var_30 = {var_28: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = module_0.Field()
    var_33 = 25
    var_34 = module_0.Field(default=var_33)
    var_35 = {var_28: var_32, var_29: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {var_28: var_6}
    var_38 = var_36.validate(var_37)
    var_39 = 'id'
    var_40 = module_0.Field()
    var_41 = module_0.Field(read_only=var_10)
    var_42 = {var_28: var_40, var_39: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {var_28: var_6}
    var_45 = var_43.validate(var_44)
    var_46 = 'details'
    var_47 = module_0.Field()
    var_48 = module_0.Field()
    var_49 = {var_29: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = {var_28: var_47, var_46: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'details'
    var_55 = 'John'
    var_56 = 'age'
    var_57 = 'invalid'
    var_58 = {var_56: var_57}
    var_59 = {var_53: var_55, var_54: var_58}
    var_60 = var_52.validate(var_59)



# Parsed testcases at query #51
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
    var_12 = 'Unknown'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_1: var_9}
    var_18 = var_16.validate(var_17)
    var_19 = module_1.Schema(var_6)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = module_1.Schema(var_6)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = 'not a dict'
    var_26 = var_7.validate(var_25)
    var_27 = 123
    var_28 = 'value'
    var_29 = {var_27: var_28}
    var_30 = var_7.validate(var_29)
    var_31 = 'age'
    var_32 = 30
    var_33 = {var_31: var_32}
    var_34 = var_7.validate(var_33)
    var_35 = 3
    var_36 = module_0.Field(allow_null=var_33)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'name'
    var_40 = 'Jo'
    var_41 = {var_39: var_40}
    var_42 = var_38.validate(var_41)



# Parsed testcases at query #52
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
    assert var_9 == 'validated_value'



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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
    var_12 = {var_0: var_8}
    var_13 = var_7.validate(var_12)
    var_14 = {var_1: var_9}
    var_15 = var_7.validate(var_14)
    var_16 = 123
    var_17 = {var_16: var_8}
    var_18 = var_7.validate(var_17)
    var_19 = None
    var_20 = var_7.validate(var_19)
    var_21 = 'not a dict'
    var_22 = var_7.validate(var_21)
    var_23 = module_1.Schema(var_6)
    var_24 = None
    var_25 = var_23.validate(var_24)
    assert var_25 is None



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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
    var_24 = module_0.Field()
    var_25 = {var_20: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = {}
    var_28 = var_26.validate(var_27)
    var_29 = 'default_name'
    var_30 = module_0.Field(default=var_29)
    var_31 = {var_27: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = module_0.Field(read_only=var_10)
    var_36 = {var_27: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = {}
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Field()
    var_41 = 'valid'
    var_42 = 'invalid'
    var_43 = module_2.Message(text=var_42, code=var_42)
    var_44 = [var_43]
    var_45 = module_2.ValidationError(messages=var_44)
    var_46 = (var_12, var_45)
    var_47 = 'nested'
    var_48 = {var_47: var_40}
    var_49 = module_1.Schema(var_48)
    var_50 = {var_47: var_41}
    var_51 = var_49.validate(var_50)
    var_52 = 'nested'
    var_53 = 'invalid'
    var_54 = {var_52: var_53}
    var_55 = var_49.validate(var_54)



# Parsed testcases at query #57
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #58
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



# Parsed testcases at query #59
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



# Parsed testcases at query #60
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



# Parsed testcases at query #61
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_4.validate(var_11)
    var_13 = module_1.Field(allow_null=var_11)
    var_14 = 'invalid'
    var_15 = var_7 / var_11
    var_16 = 'test_field_validation'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'invalid'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #62
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
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #63
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'test_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'validated_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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
    var_10 = 'test_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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
    var_9 = False
    var_10 = module_0.Reference(var_3, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_1.Field(allow_null=var_11)
    var_14 = 'invalid'
    var_15 = var_11 / var_9
    var_16 = 'test_field_validation'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'invalid'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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



# Parsed testcases at query #72
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
    var_56 = 'value'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'inner'
    var_61 = {var_60: var_59}
    var_62 = module_1.Schema(var_61)
    var_63 = 'inner'
    var_64 = 'invalid'
    var_65 = 'data'
    var_66 = {var_64: var_65}
    var_67 = {var_63: var_66}
    var_68 = var_62.validate(var_67)



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = 'test_value'
    var_12 = var_3.validate(var_11)
    assert var_12 == 'mocked_value'



# Parsed testcases at query #75
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



# Parsed testcases at query #76
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
    var_49 = module_0.Field()
    var_50 = module_0.Field()
    var_51 = {var_38: var_49, var_39: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'age'
    var_55 = 'John'
    var_56 = 'invalid'
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)



# Parsed testcases at query #77
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
    var_20 = 'invalid key'
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
    var_35 = 'DefaultName'
    var_36 = module_0.Field(default=var_35, allow_null=var_33)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field(allow_null=var_33, read_only=var_4)
    var_42 = {var_31: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = 'other'
    var_45 = 'value'
    var_46 = {var_44: var_45}
    var_47 = var_43.validate(var_46)



# Parsed testcases at query #78
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



# Parsed testcases at query #79
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = module_1.Field()
    var_12 = ()
    var_13 = 'error'
    var_14 = module_2.Message(text=var_13, code=var_13)
    var_15 = [var_14]
    var_16 = module_2.ValidationError(messages=var_15)
    var_17 = 'error_field'
    var_18 = module_0.Reference(var_17, var_0)
    var_19 = 'any_value'
    var_20 = var_18.validate(var_19)



# Parsed testcases at query #80
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
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #81
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



# Parsed testcases at query #82
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
    var_57 = 'Invalid'
    var_58 = 'invalid'
    var_59 = module_2.Message(text=var_57, code=var_58)
    var_60 = [var_59]
    var_61 = module_2.ValidationError(messages=var_60)
    var_62 = (var_14, var_61)
    var_63 = {var_38: var_56}
    var_64 = module_1.Schema(var_63)
    var_65 = 'name'
    var_66 = 'invalid_value'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #83
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



# Parsed testcases at query #84
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
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'expected'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'expected'
    var_12 = 'unexpected'
    var_13 = var_3.validate(var_12)



# Parsed testcases at query #85
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
    var_57 = 'error'
    var_58 = 'test'
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



# Parsed testcases at query #86
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
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #87
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



# Parsed testcases at query #88
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
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #89
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
    var_12 = 'DefaultName'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_1: var_9}
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Field(allow_null=var_2, read_only=var_4)
    var_20 = module_0.Field(allow_null=var_4)
    var_21 = {var_0: var_19, var_1: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = {var_1: var_9}
    var_24 = var_22.validate(var_23)
    var_25 = module_1.Schema(var_6)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = module_1.Schema(var_6)
    var_29 = None
    var_30 = var_28.validate(var_29)
    var_31 = 'not a dict'
    var_32 = var_7.validate(var_31)
    var_33 = 1
    var_34 = 'value'
    var_35 = {var_33: var_34}
    var_36 = var_7.validate(var_35)
    var_37 = 'age'
    var_38 = 30
    var_39 = {var_37: var_38}
    var_40 = var_7.validate(var_39)



# Parsed testcases at query #90
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = ()
    var_12 = 'error'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #91
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'
    var_12 = 123
    var_13 = var_3.validate(var_12)
    assert var_13 is None



# Parsed testcases at query #92
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'test error'
    var_12 = 'test'
    var_13 = module_2.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'invalid_value'
    var_17 = var_3.validate(var_16)



# Parsed testcases at query #93
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_4, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = None
    var_14 = var_5.validate(var_13)
    var_15 = 'invalid'
    var_16 = var_5.validate(var_15)



# Parsed testcases at query #94
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
    var_57 = 'Invalid'
    var_58 = 'invalid'
    var_59 = module_2.Message(text=var_57, code=var_58)
    var_60 = [var_59]
    var_61 = module_2.ValidationError(messages=var_60)
    var_62 = (var_14, var_61)
    var_63 = 'child'
    var_64 = {var_63: var_56}
    var_65 = module_1.Schema(var_64)
    var_66 = 'child'
    var_67 = 'value'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)



# Parsed testcases at query #95
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #96
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
    var_10 = 'test_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #97
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
    var_16 = 1
    var_17 = 'age'
    var_18 = 'John'
    var_19 = 30
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = var_7.validate(var_20)
    var_22 = None
    var_23 = var_7.validate(var_22)
    var_24 = module_1.Schema(var_6)
    var_25 = None
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = 'not a dict'
    var_28 = var_7.validate(var_27)
    var_29 = module_0.Field()
    var_30 = 25
    var_31 = module_0.Field(default=var_30)
    var_32 = {var_27: var_29, var_28: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {var_27: var_8}
    var_35 = var_33.validate(var_34)
    var_36 = 'id'
    var_37 = module_0.Field()
    var_38 = module_0.Field(read_only=var_18)
    var_39 = {var_27: var_37, var_36: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = {var_27: var_8}
    var_42 = var_40.validate(var_41)



# Parsed testcases at query #98
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
    var_12 = 'default_name'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(default=var_2, allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {}
    var_18 = var_16.validate(var_17)
    var_19 = 'age'
    var_20 = 30
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = None
    var_24 = var_7.validate(var_23)
    var_25 = 'not a dict'
    var_26 = var_7.validate(var_25)
    var_27 = 123
    var_28 = 'invalid key'
    var_29 = {var_27: var_28}
    var_30 = var_7.validate(var_29)
    var_31 = 'details'
    var_32 = module_0.Field(allow_null=var_29)
    var_33 = module_0.Field(allow_null=var_29)
    var_34 = {var_28: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = {var_27: var_32, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'details'
    var_40 = 'John'
    var_41 = 'age'
    var_42 = 'invalid'
    var_43 = {var_41: var_42}
    var_44 = {var_38: var_40, var_39: var_43}
    var_45 = var_37.validate(var_44)



# Parsed testcases at query #99
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
    var_30 = 'value'
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
    var_65 = 'invalid_value'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)



# Parsed testcases at query #100
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
    var_5 = module_0.Field(default=var_2, allow_null=var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = {var_0: var_8}
    var_13 = var_7.validate(var_12)
    var_14 = module_1.Schema(var_6)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = None
    var_18 = var_7.validate(var_17)
    var_19 = 'not a dict'
    var_20 = var_7.validate(var_19)
    var_21 = 123
    var_22 = 'invalid key'
    var_23 = {var_21: var_22}
    var_24 = var_7.validate(var_23)
    var_25 = 'age'
    var_26 = 30
    var_27 = {var_25: var_26}
    var_28 = var_7.validate(var_27)
    var_29 = module_0.Field(allow_null=var_27)
    var_30 = 'error'
    var_31 = 'nested error'
    var_32 = 'nested'
    var_33 = module_2.Message(text=var_31, code=var_32)
    var_34 = [var_33]
    var_35 = module_2.ValidationError(messages=var_34)
    var_36 = (var_15, var_35)
    var_37 = {var_32: var_29}
    var_38 = module_1.Schema(var_37)
    var_39 = 'nested'
    var_40 = 'error'
    var_41 = {var_39: var_40}
    var_42 = var_38.validate(var_41)



# Parsed testcases at query #101
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
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #102
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
    var_12 = {var_1: var_9}
    var_13 = var_7.validate(var_12)
    var_14 = 123
    var_15 = 'invalid'
    var_16 = {var_13: var_8, var_14: var_15}
    var_17 = var_7.validate(var_16)
    var_18 = module_1.Schema(var_6)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = 'not a dict'
    var_22 = var_18.validate(var_21)
    var_23 = module_0.Field(allow_null=var_2)
    var_24 = 25
    var_25 = module_0.Field(default=var_24, allow_null=var_4)
    var_26 = {var_21: var_23, var_22: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {var_21: var_8}
    var_29 = var_27.validate(var_28)
    var_30 = 'id'
    var_31 = module_0.Field(allow_null=var_2)
    var_32 = module_0.Field(allow_null=var_4, read_only=var_4)
    var_33 = {var_21: var_31, var_30: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {var_21: var_8}
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Field(allow_null=var_2)
    var_38 = 'details'
    var_39 = module_0.Field(allow_null=var_2)
    var_40 = {var_21: var_39, var_38: var_37}
    var_41 = module_1.Schema(var_40)
    var_42 = None
    var_43 = {var_21: var_8, var_38: var_42}
    var_44 = var_41.validate(var_43)
    var_45 = module_1.Schema(var_40)
    var_46 = var_45.validate(var_42)
    assert var_46 is None



# Parsed testcases at query #103
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
    var_49 = 3
    var_50 = module_0.Field()
    var_51 = {var_38: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'Jo'
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)



# Parsed testcases at query #104
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
    var_12 = 'Unknown'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(default=var_2, allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {}
    var_18 = var_16.validate(var_17)
    var_19 = module_1.Schema(var_6)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = None
    var_23 = var_7.validate(var_22)
    var_24 = 'not a dict'
    var_25 = var_7.validate(var_24)
    var_26 = 123
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = var_7.validate(var_28)
    var_30 = 'age'
    var_31 = 30
    var_32 = {var_30: var_31}
    var_33 = var_7.validate(var_32)
    var_34 = module_0.Field(allow_null=var_32)
    var_35 = module_0.Field(allow_null=var_32)
    var_36 = {var_30: var_34, var_31: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = 'name'
    var_39 = 'age'
    var_40 = 'John'
    var_41 = None
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = var_37.validate(var_42)



# Parsed testcases at query #105
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'error'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #106
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



# Parsed testcases at query #107
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



# Parsed testcases at query #108
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
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #109
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #110
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
    var_12 = 'Unknown'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(default=var_2, allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {}
    var_18 = var_16.validate(var_17)
    var_19 = module_1.Schema(var_6)
    var_20 = 'age'
    var_21 = 30
    var_22 = {var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = module_1.Schema(var_6)
    var_25 = None
    var_26 = var_24.validate(var_25)
    var_27 = module_1.Schema(var_6)
    var_28 = 'not a dict'
    var_29 = var_27.validate(var_28)
    var_30 = module_1.Schema(var_6)
    var_31 = 123
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'user'
    var_36 = module_0.Field(allow_null=var_33)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {var_35: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = 'user'
    var_42 = 'name'
    var_43 = None
    var_44 = {var_42: var_43}
    var_45 = {var_41: var_44}
    var_46 = var_40.validate(var_45)



# Parsed testcases at query #111
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
    var_42 = 5
    var_43 = module_0.Field()
    var_44 = {var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'name'
    var_47 = 'John'
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = 'default_name'
    var_51 = module_0.Field(default=var_50)
    var_52 = {var_46: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Field(read_only=var_12)
    var_57 = {var_46: var_56}
    var_58 = module_1.Schema(var_57)
    var_59 = {}
    var_60 = var_58.validate(var_59)



# Parsed testcases at query #112
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



# Parsed testcases at query #113
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



# Parsed testcases at query #114
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
    var_10 = ()
    var_11 = 'Invalid'
    var_12 = 'invalid_value'
    var_13 = var_3.validate(var_12)



# Parsed testcases at query #115
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



# Parsed testcases at query #116
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
    var_14 = None
    var_15 = var_5.validate(var_14)
    var_16 = 'not a dict'
    var_17 = var_5.validate(var_16)
    var_18 = 1
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = var_5.validate(var_20)
    var_22 = 'name'
    var_23 = 'John'
    var_24 = {var_22: var_23}
    var_25 = var_5.validate(var_24)
    var_26 = module_0.Field()
    var_27 = 25
    var_28 = module_0.Field(default=var_27)
    var_29 = {var_22: var_26, var_23: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = {var_22: var_6}
    var_32 = var_30.validate(var_31)
    var_33 = 'id'
    var_34 = module_0.Field()
    var_35 = module_0.Field(read_only=var_10)
    var_36 = {var_22: var_34, var_33: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = {var_22: var_6}
    var_39 = var_37.validate(var_38)
    var_40 = 'details'
    var_41 = module_0.Field()
    var_42 = module_0.Field()
    var_43 = {var_23: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {var_22: var_41, var_40: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = 'name'
    var_48 = 'details'
    var_49 = 'John'
    var_50 = 'age'
    var_51 = 'invalid'
    var_52 = {var_50: var_51}
    var_53 = {var_47: var_49, var_48: var_52}
    var_54 = var_46.validate(var_53)



# Parsed testcases at query #117
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = True
    var_5 = module_0.Field(default=var_2, allow_null=var_4)
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
    var_27 = {var_23: var_8}
    var_28 = var_7.validate(var_27)
    var_29 = 'id'
    var_30 = module_0.Field(allow_null=var_25)
    var_31 = module_0.Field(read_only=var_4)
    var_32 = {var_23: var_30, var_29: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {var_23: var_8}
    var_35 = var_33.validate(var_34)



# Parsed testcases at query #118
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
    var_42 = 'default_name'
    var_43 = module_0.Field(default=var_42)
    var_44 = {var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Field(read_only=var_12)
    var_49 = module_0.Field()
    var_50 = {var_38: var_48, var_39: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_39: var_7}
    var_53 = var_51.validate(var_52)
    var_54 = 'value'
    var_55 = module_0.Field()
    var_56 = {var_54: var_55}
    var_57 = module_1.Schema(var_56)
    var_58 = 'inner'
    var_59 = {var_58: var_57}
    var_60 = module_1.Schema(var_59)
    var_61 = 'inner'
    var_62 = 'invalid'
    var_63 = 'data'
    var_64 = {var_62: var_63}
    var_65 = {var_61: var_64}
    var_66 = var_60.validate(var_65)



# Parsed testcases at query #119
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
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'not a dict'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #120
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
    var_13 = 123
    var_14 = {var_11: var_13}
    var_15 = var_5.validate(var_14)



# Parsed testcases at query #121
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



# Parsed testcases at query #122
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = 'type'
    var_11 = 'Invalid type'
    var_12 = ()
    var_13 = module_2.Message(text=var_11, code=var_10)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'invalid_value'
    var_17 = var_3.validate(var_16)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_12 = {var_0: var_6}
    var_13 = var_5.serialize(var_12)
    var_14 = 'address'
    var_15 = module_0.Field()
    var_16 = 'city'
    var_17 = module_0.Field()
    var_18 = {var_16: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = {var_0: var_15, var_14: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'NYC'
    var_23 = {var_16: var_22}
    var_24 = {var_0: var_6, var_14: var_23}
    var_25 = var_21.serialize(var_24)
    var_26 = var_5.serialize(var_24)



# Parsed testcases at query #2
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
    var_9 = var_5.serialize(var_8)
    var_10 = None
    var_11 = var_5.serialize(var_10)
    assert var_11 is None
    var_12 = 'Jane'
    var_13 = 25
    var_14 = {var_0: var_6}
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



# Parsed testcases at query #4
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_7.validate(var_18)
    var_20 = 123
    var_21 = 'invalid key'
    var_22 = {var_20: var_21}
    var_23 = var_7.validate(var_22)
    var_24 = module_0.Field(allow_null=var_22)
    var_25 = {var_20: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 'age'
    var_28 = 30
    var_29 = {var_27: var_28}
    var_30 = var_26.validate(var_29)
    var_31 = 'default_name'
    var_32 = module_0.Field(default=var_31, allow_null=var_29)
    var_33 = {var_27: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Field(allow_null=var_29)
    var_38 = module_0.Field(allow_null=var_29)
    var_39 = {var_27: var_37, var_28: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = 'name'
    var_42 = 'age'
    var_43 = 'John'
    var_44 = 'invalid'
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = var_40.validate(var_45)



# Parsed testcases at query #5
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
    var_28 = 'Invalid age'
    var_29 = 'invalid'
    var_30 = module_2.Message(text=var_28, code=var_29)
    var_31 = [var_30]
    var_32 = module_2.ValidationError(messages=var_31)
    var_33 = (var_13, var_32)
    var_34 = {var_24: var_27}
    var_35 = module_1.Schema(var_34)
    var_36 = 'age'
    var_37 = -5
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)



# Parsed testcases at query #6
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
    var_13 = 'not a dict'
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #7
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_7.validate(var_18)
    var_20 = 123
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_7.validate(var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_7.validate(var_26)
    var_28 = 'Name must be John'
    var_29 = lambda x: x == var_8 or var_28
    var_30 = [var_29]
    var_31 = module_0.Field(allow_null=var_26)
    var_32 = {var_24: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'name'
    var_35 = 'Jane'
    var_36 = {var_34: var_35}
    var_37 = var_33.validate(var_36)
    var_38 = 'DefaultName'
    var_39 = module_0.Field(default=var_38)
    var_40 = {var_34: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Field(read_only=var_4)
    var_45 = module_0.Field()
    var_46 = {var_34: var_44, var_35: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {var_35: var_9}
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #8
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
    var_19 = 1
    var_20 = 'invalid key'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = 'invalid'
    var_28 = lambda x: x != var_27
    var_29 = [var_28]
    var_30 = module_0.Field(allow_null=var_25)
    var_31 = {var_23: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = 'name'
    var_34 = 'invalid'
    var_35 = {var_33: var_34}
    var_36 = var_32.validate(var_35)
    var_37 = 'default'
    var_38 = module_0.Field(default=var_37, allow_null=var_35)
    var_39 = {var_33: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = module_0.Field(allow_null=var_35, read_only=var_4)
    var_44 = {var_33: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)



# Parsed testcases at query #9
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
    var_13 = 'invalid_key'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_5.validate(var_15)



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
    var_57 = 'name'
    var_58 = 'age'
    var_59 = 'John'
    var_60 = -5
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = var_53.validate(var_61)



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
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #12
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'invalid_key'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_5.validate(var_15)



# Parsed testcases at query #13
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #14
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
    var_28 = 'Invalid age'
    var_29 = 'invalid'
    var_30 = module_2.Message(text=var_28, code=var_29)
    var_31 = [var_30]
    var_32 = module_2.ValidationError(messages=var_31)
    var_33 = (var_13, var_32)
    var_34 = {var_24: var_27}
    var_35 = module_1.Schema(var_34)
    var_36 = 'age'
    var_37 = -5
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = 'default_value'
    var_41 = module_0.Field(default=var_40)
    var_42 = {var_36: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)



# Parsed testcases at query #15
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
    var_42 = 'default_name'
    var_43 = module_0.Field(default=var_42)
    var_44 = {var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)
    var_48 = 'id'
    var_49 = module_0.Field()
    var_50 = module_0.Field(read_only=var_12)
    var_51 = {var_38: var_49, var_48: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_38: var_6}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Field()
    var_56 = 'error'
    var_57 = 'child_error'
    var_58 = module_2.Message(text=var_56, code=var_57)
    var_59 = [var_58]
    var_60 = module_2.ValidationError(messages=var_59)
    var_61 = (var_14, var_60)
    var_62 = 'child'
    var_63 = {var_62: var_55}
    var_64 = module_1.Schema(var_63)
    var_65 = 'child'
    var_66 = 'value'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #16
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
    var_19 = 1
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = lambda x: x == var_8
    var_28 = [var_27]
    var_29 = module_0.Field(allow_null=var_25)
    var_30 = {var_23: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'name'
    var_33 = 'Jane'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 'DefaultName'
    var_37 = module_0.Field(default=var_36, allow_null=var_34)
    var_38 = {var_32: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field(allow_null=var_34, read_only=var_4)
    var_43 = {var_32: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {}
    var_46 = var_44.validate(var_45)



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



# Parsed testcases at query #18
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



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
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #20
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'invalid_key'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_5.validate(var_15)



# Parsed testcases at query #21
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
    var_56 = 'value'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'inner'
    var_61 = {var_60: var_59}
    var_62 = module_1.Schema(var_61)
    var_63 = 'inner'
    var_64 = 'value'
    var_65 = None
    var_66 = {var_64: var_65}
    var_67 = {var_63: var_66}
    var_68 = var_62.validate(var_67)



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
    var_6 = module_0.Reference(var_5, var_0)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'not a dict'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'test'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = module_0.Reference(var_8, var_0)
    var_17 = 'invalid_value'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    var_9 = True
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = None
    var_14 = var_4.validate(var_13)



# Parsed testcases at query #25
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_15.validate(var_18)
    var_20 = 1
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_15.validate(var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_15.validate(var_26)
    var_28 = module_0.Field(allow_null=var_26)
    var_29 = module_0.Field(allow_null=var_26)
    var_30 = {var_24: var_28, var_25: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'name'
    var_33 = 'age'
    var_34 = 'John'
    var_35 = None
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = var_31.validate(var_36)
    var_38 = module_0.Field(allow_null=var_34)
    var_39 = 25
    var_40 = module_0.Field(default=var_39, allow_null=var_36)
    var_41 = {var_32: var_38, var_33: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = {var_32: var_8}
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Field(allow_null=var_34)
    var_46 = module_0.Field(allow_null=var_36, read_only=var_36)
    var_47 = {var_32: var_45, var_33: var_46}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_32: var_8}
    var_50 = var_48.validate(var_49)



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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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
    var_48 = module_0.Field(read_only=var_12)
    var_49 = module_0.Field()
    var_50 = {var_42: var_48, var_43: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_43: var_7}
    var_53 = var_51.validate(var_52)



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_22 = 'not a dict'
    var_23 = var_5.validate(var_22)
    var_24 = 1
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_5.validate(var_26)
    var_28 = module_0.Field()
    var_29 = module_0.Field()
    var_30 = {var_24: var_28, var_25: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'name'
    var_33 = 'John'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 'default_name'
    var_37 = module_0.Field(default=var_36)
    var_38 = {var_32: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = 'id'
    var_43 = module_0.Field()
    var_44 = module_0.Field(read_only=var_12)
    var_45 = {var_32: var_43, var_42: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_32: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = 'user'
    var_50 = module_0.Field()
    var_51 = {var_32: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = {var_49: var_52}
    var_54 = module_1.Schema(var_53)
    var_55 = 'user'
    var_56 = 'invalid_key'
    var_57 = 'value'
    var_58 = {var_56: var_57}
    var_59 = {var_55: var_58}
    var_60 = var_54.validate(var_59)



# Parsed testcases at query #31
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = None
    var_8 = var_4.validate(var_7)
    var_9 = 'test_value'
    var_10 = var_4.validate(var_9)
    assert var_10 == 'validated_value'
    var_11 = 'error'
    var_12 = module_2.Message(text=var_11, code=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = 'invalid_value'
    var_16 = var_4.validate(var_15)



# Parsed testcases at query #32
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
    var_10 = 'invalid'
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'invalid'
    var_15 = var_3.validate(var_14)



# Parsed testcases at query #33
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
    var_70 = 'Main St'
    var_71 = {var_69: var_70}
    var_72 = {var_66: var_68, var_67: var_71}
    var_73 = var_65.validate(var_72)



# Parsed testcases at query #34
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
    var_49 = module_0.Field()
    var_50 = module_0.Field()
    var_51 = {var_38: var_49, var_39: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'age'
    var_55 = 'John'
    var_56 = 'invalid age'
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)
    var_59 = 'id'
    var_60 = module_0.Field()
    var_61 = module_0.Field(read_only=var_12)
    var_62 = {var_53: var_60, var_59: var_61}
    var_63 = module_1.Schema(var_62)
    var_64 = {var_53: var_58}
    var_65 = var_63.validate(var_64)



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)



# Parsed testcases at query #38
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
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = ()
    var_12 = 'Test error'
    var_13 = 'error_field'
    var_14 = module_0.Reference(var_13, var_0)
    var_15 = 'test_value'
    var_16 = var_14.validate(var_15)



# Parsed testcases at query #39
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



# Parsed testcases at query #40
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
    var_13 = 'invalid_key'
    var_14 = 123
    var_15 = {var_13: var_14}
    var_16 = var_5.validate(var_15)



# Parsed testcases at query #41
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
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #42
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



# Parsed testcases at query #43
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
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
    var_11 = 'expected'
    var_12 = 'test_field_validation'
    var_13 = module_0.Reference(var_12, var_0)
    var_14 = var_13.validate(var_11)
    assert var_14 == 'expected'
    var_15 = 'unexpected'
    var_16 = var_13.validate(var_15)



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_1.Field()
    var_11 = ()
    var_12 = 'error'
    var_13 = module_2.Message(text=var_12, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'error_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'any_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #52
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid'
    var_5 = var_3.validate(var_4)
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



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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
    var_9 = False
    var_10 = module_0.Reference(var_3, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_1.Field(allow_null=var_9)
    var_14 = 'test_field_error'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = None
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #55
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'test'
    var_12 = module_2.Message(text=var_11, code=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = 'invalid_value'
    var_16 = var_3.validate(var_15)



# Parsed testcases at query #56
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



# Parsed testcases at query #57
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
    var_42 = 'default_name'
    var_43 = module_0.Field(default=var_42)
    var_44 = {var_38: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = {}
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Field(read_only=var_12)
    var_49 = {var_38: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = 'city'
    var_54 = module_0.Field()
    var_55 = {var_53: var_54}
    var_56 = module_1.Schema(var_55)
    var_57 = 'address'
    var_58 = module_0.Field()
    var_59 = {var_38: var_58, var_57: var_56}
    var_60 = module_1.Schema(var_59)
    var_61 = 'name'
    var_62 = 'address'
    var_63 = 'John'
    var_64 = 'city'
    var_65 = 123
    var_66 = {var_64: var_65}
    var_67 = {var_61: var_63, var_62: var_66}
    var_68 = var_60.validate(var_67)



# Parsed testcases at query #58
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
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #59
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



# Parsed testcases at query #60
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
    var_51 = 'error'
    var_52 = module_2.Message(text=var_51, code=var_51)
    var_53 = [var_52]
    var_54 = module_2.ValidationError(messages=var_53)
    var_55 = (var_14, var_54)
    var_56 = {var_37: var_50}
    var_57 = module_1.Schema(var_56)
    var_58 = 'name'
    var_59 = 'test'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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
    var_48 = module_0.Field(read_only=var_12)
    var_49 = module_0.Field()
    var_50 = {var_42: var_48, var_43: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = {var_43: var_7}
    var_53 = var_51.validate(var_52)



# Parsed testcases at query #63
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
    var_55 = 123
    var_56 = {var_37: var_6, var_48: var_55}
    var_57 = var_52.validate(var_56)
    var_58 = 'age'
    var_59 = -5
    var_60 = {var_58: var_59}
    var_61 = var_52.validate(var_60)



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    assert var_12 == 'mocked_value'



# Parsed testcases at query #68
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
    var_24 = module_0.Field()
    var_25 = {var_23: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = 123
    var_28 = {var_27: var_7}
    var_29 = var_26.validate(var_28)
    var_30 = module_0.Field()
    var_31 = {var_29: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = 'not a dict'
    var_34 = var_32.validate(var_33)
    var_35 = module_0.Field()
    var_36 = {var_34: var_35}
    var_37 = module_1.Schema(var_36)
    var_38 = None
    var_39 = var_37.validate(var_38)
    assert var_39 is None
    var_40 = module_0.Field()
    var_41 = {var_34: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = None
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Field()
    var_46 = 25
    var_47 = module_0.Field(default=var_46)
    var_48 = {var_44: var_45, var_1: var_47}
    var_49 = module_1.Schema(var_48)
    var_50 = {var_44: var_7}
    var_51 = var_49.validate(var_50)
    var_52 = 'street'
    var_53 = 'city'
    var_54 = module_0.Field()
    var_55 = module_0.Field()
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = 'address'
    var_58 = module_0.Field()
    var_59 = module_1.Schema(var_56)
    var_60 = {var_44: var_58, var_57: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = '123 Main St'
    var_63 = 'New York'
    var_64 = {var_52: var_62, var_53: var_63}
    var_65 = {var_44: var_7, var_57: var_64}
    var_66 = var_61.validate(var_65)
    var_67 = module_0.Field()
    var_68 = module_0.Field()
    var_69 = {var_52: var_67, var_53: var_68}
    var_70 = module_0.Field()
    var_71 = module_1.Schema(var_69)
    var_72 = {var_44: var_70, var_57: var_71}
    var_73 = module_1.Schema(var_72)
    var_74 = {var_52: var_62}
    var_75 = {var_44: var_7, var_57: var_74}
    var_76 = var_73.validate(var_75)



# Parsed testcases at query #69
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
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = None
    var_11 = var_6.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_6.validate(var_12)
    var_14 = 'invalid'
    var_15 = var_6.validate(var_14)



# Parsed testcases at query #70
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
    var_9 = False
    var_10 = module_0.Reference(var_3, var_0)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_1.Field(allow_null=var_9)
    var_14 = 'valid'
    var_15 = var_11 / var_9
    var_16 = 'strict_field'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'invalid_value'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #71
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'test_error'
    var_12 = 'test_code'
    var_13 = module_2.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'invalid_value'
    var_17 = var_3.validate(var_16)



# Parsed testcases at query #72
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
    var_10 = 'invalid'
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'invalid'
    var_15 = var_3.validate(var_14)



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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
    var_57 = 'Invalid'
    var_58 = 'invalid'
    var_59 = module_2.Message(text=var_57, code=var_58)
    var_60 = [var_59]
    var_61 = module_2.ValidationError(messages=var_60)
    var_62 = (var_14, var_61)
    var_63 = {var_38: var_56}
    var_64 = module_1.Schema(var_63)
    var_65 = 'name'
    var_66 = 'invalid_value'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #75
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
    var_56 = 123
    var_57 = {var_38: var_6, var_49: var_56}
    var_58 = var_53.validate(var_57)
    var_59 = module_0.Field()
    var_60 = 'Invalid'
    var_61 = 'invalid'
    var_62 = module_2.Message(text=var_60, code=var_61)
    var_63 = [var_62]
    var_64 = module_2.ValidationError(messages=var_63)
    var_65 = (var_14, var_64)
    var_66 = {var_38: var_59}
    var_67 = module_1.Schema(var_66)
    var_68 = 'name'
    var_69 = 'invalid'
    var_70 = {var_68: var_69}
    var_71 = var_67.validate(var_70)



# Parsed testcases at query #76
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = True
    var_3 = module_1.Field(allow_null=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_0.Schema(var_4)
    var_6 = 'test_schema'
    var_7 = module_0.Reference(var_6, var_0)
    var_8 = 'test'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = None
    var_12 = var_7.validate(var_11)
    assert var_12 is None
    var_13 = None
    var_14 = var_7.validate(var_13)
    var_15 = 'invalid'
    var_16 = var_7.validate(var_15)



# Parsed testcases at query #77
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
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)
    var_11 = ()
    var_12 = 'test error'
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #78
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
    var_7 = 'person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    assert var_14 is None
    var_15 = None
    var_16 = var_8.validate(var_15)
    var_17 = 'not a dict'
    var_18 = var_8.validate(var_17)
    var_19 = 1
    var_20 = 'invalid key'
    var_21 = {var_19: var_20}
    var_22 = var_8.validate(var_21)
    var_23 = 'name'
    var_24 = 'John'
    var_25 = {var_23: var_24}
    var_26 = var_8.validate(var_25)



# Parsed testcases at query #79
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
    var_19 = 1
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = module_0.Field(allow_null=var_25)
    var_28 = module_0.Field(allow_null=var_25)
    var_29 = {var_23: var_27, var_24: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'name'
    var_32 = 'age'
    var_33 = 'John'
    var_34 = -1
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = var_30.validate(var_35)
    var_37 = module_0.Field(allow_null=var_33)
    var_38 = module_0.Field(default=var_33, allow_null=var_35)
    var_39 = {var_31: var_37, var_32: var_38}
    var_40 = module_1.Schema(var_39)
    var_41 = {var_31: var_8}
    var_42 = var_40.validate(var_41)
    var_43 = 'id'
    var_44 = module_0.Field(allow_null=var_33)
    var_45 = module_0.Field(allow_null=var_33, read_only=var_35)
    var_46 = {var_31: var_44, var_43: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = {var_31: var_8}
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #80
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



# Parsed testcases at query #81
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
    var_51 = 'child error'
    var_52 = 'child_error'
    var_53 = module_2.Message(text=var_51, code=var_52)
    var_54 = [var_53]
    var_55 = module_2.ValidationError(messages=var_54)
    var_56 = (var_14, var_55)
    var_57 = 'child'
    var_58 = {var_57: var_50}
    var_59 = module_1.Schema(var_58)
    var_60 = 'child'
    var_61 = 'invalid'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)



# Parsed testcases at query #82
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
    var_10 = 'invalid'
    var_11 = 1
    var_12 = 0
    var_13 = var_11 / var_12
    var_14 = 'invalid'
    var_15 = var_3.validate(var_14)



# Parsed testcases at query #83
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = module_1.Field(allow_null=var_1)
    var_3 = 'test_field'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = var_4.validate(var_7)
    var_9 = None
    var_10 = var_4.validate(var_9)
    assert var_10 is None
    var_11 = 'invalid_value'
    var_12 = var_4.validate(var_11)



# Parsed testcases at query #84
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'valid_input'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_input'
    var_6 = None
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #85
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



# Parsed testcases at query #86
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



# Parsed testcases at query #87
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'error'
    var_12 = 'test'
    var_13 = module_2.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.ValidationError(messages=var_14)
    var_16 = 'invalid_value'
    var_17 = var_3.validate(var_16)



# Parsed testcases at query #88
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
    var_12 = 'Unknown'
    var_13 = module_0.Field(default=var_12, allow_null=var_2)
    var_14 = module_0.Field(allow_null=var_4)
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_1: var_9}
    var_18 = var_16.validate(var_17)
    var_19 = 'id'
    var_20 = module_0.Field(allow_null=var_2)
    var_21 = module_0.Field(allow_null=var_2, read_only=var_4)
    var_22 = {var_0: var_20, var_19: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_0: var_8}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Schema(var_6)
    var_27 = None
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = None
    var_30 = var_7.validate(var_29)
    var_31 = 'not a dict'
    var_32 = var_7.validate(var_31)
    var_33 = 123
    var_34 = 'invalid key'
    var_35 = {var_33: var_34}
    var_36 = var_7.validate(var_35)
    var_37 = 'age'
    var_38 = 30
    var_39 = {var_37: var_38}
    var_40 = var_7.validate(var_39)
    var_41 = 3
    var_42 = module_0.Field(allow_null=var_39)
    var_43 = {var_37: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = 'name'
    var_46 = 'Jo'
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)



# Parsed testcases at query #89
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



# Parsed testcases at query #90
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
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = None
    var_10 = var_5.validate(var_9)
    assert var_10 is None
    var_11 = None
    var_12 = var_5.validate(var_11)
    var_13 = 'invalid'
    var_14 = var_5.validate(var_13)



# Parsed testcases at query #91
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
    var_13 = 123
    var_14 = {var_11: var_13}
    var_15 = var_5.validate(var_14)



# Parsed testcases at query #92
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



# Parsed testcases at query #93
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
    var_13 = 'value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #94
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
    var_6 = var_1.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_3.validate(var_9)



# Parsed testcases at query #95
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



# Parsed testcases at query #96
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
    var_13 = 'any_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #97
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
    var_20 = 123
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_15.validate(var_22)
    var_24 = module_0.Field()
    var_25 = module_0.Field()
    var_26 = {var_20: var_24, var_21: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = 'name'
    var_29 = 'John'
    var_30 = {var_28: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = module_0.Field()
    var_33 = 25
    var_34 = module_0.Field(default=var_33)
    var_35 = {var_28: var_32, var_29: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {var_28: var_6}
    var_38 = var_36.validate(var_37)
    var_39 = 'details'
    var_40 = module_0.Field()
    var_41 = module_0.Field()
    var_42 = {var_29: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {var_28: var_40, var_39: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'name'
    var_47 = 'details'
    var_48 = 'John'
    var_49 = 'age'
    var_50 = 'invalid'
    var_51 = {var_49: var_50}
    var_52 = {var_46: var_48, var_47: var_51}
    var_53 = var_45.validate(var_52)



# Parsed testcases at query #98
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
    var_10 = False
    var_11 = 'test_value'
    var_12 = var_3.validate(var_11)



# Parsed testcases at query #99
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



# Parsed testcases at query #100
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



# Parsed testcases at query #101
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



# Parsed testcases at query #102
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #103
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_10 = ()
    var_11 = 'test'
    var_12 = module_2.Message(text=var_11, code=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = 'invalid_value'
    var_16 = var_3.validate(var_15)



# Parsed testcases at query #104
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



# Parsed testcases at query #105
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



# Parsed testcases at query #106
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
    var_56 = 3
    var_57 = module_0.Field()
    var_58 = {var_38: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'name'
    var_61 = 'Jo'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)
    var_64 = module_0.Field()
    var_65 = 18
    var_66 = module_0.Field()
    var_67 = {var_60: var_64, var_61: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = 'name'
    var_70 = 'age'
    var_71 = 'Jo'
    var_72 = 17
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = var_68.validate(var_73)



# Parsed testcases at query #107
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
    var_10 = 'test_value'
    var_11 = var_3.validate(var_10)



# Parsed testcases at query #108
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



# Parsed testcases at query #109
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
    var_31 = 'invalid key'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Field()
    var_35 = {var_30: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 5
    var_40 = module_0.Field()
    var_41 = {var_37: var_40}
    var_42 = module_1.Schema(var_41)
    var_43 = 'name'
    var_44 = 'John'
    var_45 = {var_43: var_44}
    var_46 = var_42.validate(var_45)
    var_47 = 'Default'
    var_48 = module_0.Field(default=var_47)
    var_49 = {var_43: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Field(read_only=var_12)
    var_54 = {var_43: var_53}
    var_55 = module_1.Schema(var_54)
    var_56 = {var_43: var_6}
    var_57 = var_55.validate(var_56)



# Parsed testcases at query #110
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



# Parsed testcases at query #111
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



# Parsed testcases at query #112
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
    var_11 = 'invalid'
    var_12 = 0
    var_13 = var_9 / var_12
    var_14 = 'invalid'
    var_15 = var_4.validate(var_14)



# Parsed testcases at query #113
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



# Parsed testcases at query #114
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

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
    var_11 = ()
    var_12 = 'Invalid'
    var_13 = 'invalid'
    var_14 = module_2.Message(text=var_12, code=var_13)
    var_15 = [var_14]
    var_16 = module_2.ValidationError(messages=var_15)
    var_17 = 'invalid_value'
    var_18 = var_4.validate(var_17)



# Parsed testcases at query #115
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



# Parsed testcases at query #116
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
    var_13 = 'invalid_value'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #117
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



# Parsed testcases at query #118
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



# Parsed testcases at query #119
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
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'TEST'



# Parsed testcases at query #120
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



# Parsed testcases at query #121
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



# Parsed testcases at query #122
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
    var_56 = 'value'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'inner'
    var_61 = {var_60: var_59}
    var_62 = module_1.Schema(var_61)
    var_63 = 'inner'
    var_64 = 'invalid'
    var_65 = 'data'
    var_66 = {var_64: var_65}
    var_67 = {var_63: var_66}
    var_68 = var_62.validate(var_67)



# Parsed testcases at query #123
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
    var_20 = 'invalid key'
    var_21 = {var_19: var_20}
    var_22 = var_7.validate(var_21)
    var_23 = 'age'
    var_24 = 30
    var_25 = {var_23: var_24}
    var_26 = var_7.validate(var_25)
    var_27 = 'default_value'
    var_28 = module_0.Field(default=var_27, allow_null=var_25)
    var_29 = {var_23: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'name'
    var_34 = 'test'
    var_35 = {var_33: var_34}
    var_36 = module_0.Field(read_only=var_4)
    var_37 = module_0.Field()
    var_38 = {var_33: var_36, var_34: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {var_34: var_9}
    var_41 = var_39.validate(var_40)



# Parsed testcases at query #124
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



# Parsed testcases at query #125
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
    var_42 = 'id'
    var_43 = module_0.Field()
    var_44 = module_0.Field(read_only=var_12)
    var_45 = {var_38: var_43, var_42: var_44}
    var_46 = module_1.Schema(var_45)
    var_47 = {var_38: var_6}
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Field()
    var_50 = 25
    var_51 = module_0.Field(default=var_50)
    var_52 = {var_38: var_49, var_39: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = {var_38: var_6}
    var_55 = var_53.validate(var_54)
    var_56 = 'street'
    var_57 = module_0.Field()
    var_58 = {var_56: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = 'address'
    var_61 = module_0.Field()
    var_62 = {var_38: var_61, var_60: var_59}
    var_63 = module_1.Schema(var_62)
    var_64 = 'name'
    var_65 = 'address'
    var_66 = 'John'
    var_67 = 'city'
    var_68 = 'NYC'
    var_69 = {var_67: var_68}
    var_70 = {var_64: var_66, var_65: var_69}
    var_71 = var_63.validate(var_70)



# Parsed testcases at query #126
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
    var_15 = module_1.Schema(var_6)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'not a dict'
    var_19 = var_15.validate(var_18)
    var_20 = 123
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_15.validate(var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_15.validate(var_26)
    var_28 = module_0.Field(allow_null=var_26)
    var_29 = {var_24: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'name'
    var_32 = None
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'default_name'
    var_36 = module_0.Field(default=var_35)
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field(default=var_35, read_only=var_4)
    var_42 = {var_31: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)



# Parsed testcases at query #127
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
    var_10 = 'Unknown'
    var_11 = module_0.Field(default=var_10)
    var_12 = module_0.Field()
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = {var_1: var_7}
    var_16 = var_14.validate(var_15)
    var_17 = module_1.Schema(var_4)
    var_18 = 'name'
    var_19 = 'John'
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = 123
    var_23 = 'age'
    var_24 = 'John'
    var_25 = 30
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = var_17.validate(var_26)
    var_28 = True
    var_29 = module_1.Schema(var_4)
    var_30 = None
    var_31 = var_29.validate(var_30)
    assert var_31 is None
    var_32 = None
    var_33 = var_17.validate(var_32)
    var_34 = 'not a dict'
    var_35 = var_17.validate(var_34)



