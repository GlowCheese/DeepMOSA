####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_9 = 'false'
    var_10 = var_0.validate(var_9)
    assert var_10 is False
    var_11 = 'on'
    var_12 = var_0.validate(var_11)
    assert var_12 is True
    var_13 = 'off'
    var_14 = var_0.validate(var_13)
    assert var_14 is False
    var_15 = '1'
    var_16 = var_0.validate(var_15)
    assert var_16 is True
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
    var_23 = 'not_a_boolean'
    var_24 = var_0.validate(var_23)
    var_25 = None
    var_26 = var_0.validate(var_25)
    var_27 = module_0.Boolean(coerce_types=var_3)
    var_28 = var_27.validate(var_25)
    assert var_28 is True
    var_29 = var_27.validate(var_3)
    assert var_29 is False
    var_30 = 'true'
    var_31 = var_27.validate(var_30)
    var_32 = module_0.Boolean()
    var_33 = None
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = var_32.validate(var_21)
    assert var_35 is None
    var_36 = 'null'
    var_37 = var_32.validate(var_36)
    assert var_37 is None
    var_38 = 'none'
    var_39 = var_32.validate(var_38)
    assert var_39 is None
    var_40 = module_0.Boolean()
    var_41 = None
    var_42 = var_40.validate(var_41)



# Parsed testcases at query #2
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
    var_9 = 'false'
    var_10 = var_0.validate(var_9)
    assert var_10 is False
    var_11 = 'on'
    var_12 = var_0.validate(var_11)
    assert var_12 is True
    var_13 = 'off'
    var_14 = var_0.validate(var_13)
    assert var_14 is False
    var_15 = '1'
    var_16 = var_0.validate(var_15)
    assert var_16 is True
    var_17 = '0'
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = ''
    var_20 = var_0.validate(var_19)
    assert var_20 is False
    var_21 = var_0.validate(var_1)
    assert var_21 is True
    var_22 = var_0.validate(var_3)
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
    var_32 = var_25.validate(var_19)
    assert var_32 is None
    var_33 = 'not-a-boolean'
    var_34 = var_0.validate(var_33)
    var_35 = 2
    var_36 = var_0.validate(var_35)
    var_37 = module_0.Boolean(coerce_types=var_3)
    var_38 = var_37.validate(var_35)
    assert var_38 is True
    var_39 = var_37.validate(var_3)
    assert var_39 is False
    var_40 = 'true'
    var_41 = var_37.validate(var_40)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    var_2 = 'hello'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'hello'
    var_5 = 42
    var_6 = module_0.Field(default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 42
    var_8 = None
    var_9 = module_0.Field(default=var_8)
    var_10 = var_9.get_default_value()
    assert var_10 is None
    var_11 = 100
    var_12 = lambda : var_11
    var_13 = module_0.Field(default=var_12)
    var_14 = var_13.get_default_value()
    assert var_14 == 100



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Const(var_0)
    var_2 = 'hello'
    var_3 = module_0.Const(var_2)
    var_4 = True
    var_5 = module_0.Const(var_4)
    var_6 = None
    var_7 = module_0.Const(var_6)
    var_8 = 10
    var_9 = True
    var_10 = module_0.Const(var_8)
    var_11 = 'fixed'
    var_12 = module_0.Const(var_11)
    var_13 = var_12.validate(var_11)
    assert var_13 == 'fixed'
    var_14 = 'wrong'
    var_15 = var_12.validate(var_14)
    var_16 = module_0.Const(var_6)
    var_17 = var_16.validate(var_6)
    assert var_17 is None
    var_18 = 'not_none'
    var_19 = var_16.validate(var_18)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
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
    var_7 = 'False'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = 'off'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = '1'
    var_14 = var_0.validate(var_13)
    assert var_14 is True
    var_15 = '0'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = ''
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = var_0.validate(var_1)
    assert var_19 is True
    var_20 = var_0.validate(var_3)
    assert var_20 is False
    var_21 = None
    var_22 = var_0.validate(var_21)
    var_23 = module_0.Boolean()
    var_24 = None
    var_25 = var_23.validate(var_24)
    assert var_25 is None
    var_26 = 'null'
    var_27 = var_23.validate(var_26)
    assert var_27 is None
    var_28 = 'none'
    var_29 = var_23.validate(var_28)
    assert var_29 is None
    var_30 = var_23.validate(var_17)
    assert var_30 is None
    var_31 = 'null'
    var_32 = var_0.validate(var_31)
    var_33 = module_0.Boolean(coerce_types=var_3)
    var_34 = var_33.validate(var_31)
    assert var_34 is True
    var_35 = 'true'
    var_36 = var_33.validate(var_35)
    var_37 = 'maybe'
    var_38 = var_0.validate(var_37)
    var_39 = True
    var_40 = [var_39]
    var_41 = var_0.validate(var_40)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = 'err'
    var_2 = 'type'
    var_3 = module_0.Message(text=var_1, code=var_2)
    var_4 = [var_3]
    var_5 = module_0.ValidationError(messages=var_4)
    var_6 = 'input'
    var_7 = None
    var_8 = 'allow_null'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = False
    var_12 = {var_8: var_11}
    var_13 = None
    var_14 = 'type error'
    var_15 = module_0.Message(text=var_14, code=var_2)
    var_16 = [var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = 'bad_input'
    var_19 = 'too small'
    var_20 = 'minimum'
    var_21 = module_0.Message(text=var_19, code=var_20)
    var_22 = [var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'input'



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Array()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Array()
    var_9 = 'not a list'
    var_10 = var_8.validate(var_9)
    var_11 = 2
    var_12 = module_0.Array(min_items=var_11)
    var_13 = 1
    var_14 = [var_13]
    var_15 = var_12.validate(var_14)
    var_16 = 3
    var_17 = module_0.Array(exact_items=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = [var_18, var_19]
    var_21 = var_17.validate(var_20)
    var_22 = module_0.Array(max_items=var_18)
    var_23 = 1
    var_24 = 2
    var_25 = [var_23, var_24]
    var_26 = var_22.validate(var_25)
    var_27 = module_0.Array(min_items=var_23)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = [var_28, var_11, var_16]
    var_31 = 1
    var_32 = 2
    var_33 = [var_31, var_32]
    var_34 = 10
    var_35 = 'hello'
    var_36 = [var_34, var_35]
    var_37 = 'not_int'
    var_38 = 'hello'
    var_39 = [var_37, var_38]
    var_40 = [var_37, var_11, var_16]
    var_41 = 1
    var_42 = 2
    var_43 = [var_41, var_41, var_42]
    var_44 = 'extra'
    var_45 = [var_41, var_44]
    var_46 = 1
    var_47 = 2.5
    var_48 = [var_46, var_47]



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 'a'
    var_6 = [var_4, var_5]
    var_7 = True
    var_8 = module_0.Array()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Array()
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Array()
    var_16 = 'not a list'
    var_17 = var_15.validate(var_16)
    var_18 = 2
    var_19 = module_0.Array(min_items=var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Array(min_items=var_7)
    var_24 = []
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Array(max_items=var_18)
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = var_26.validate(var_30)
    var_32 = module_0.Array(exact_items=var_18)
    var_33 = [var_7, var_18]
    var_34 = var_32.validate(var_33)
    var_35 = 1
    var_36 = [var_35]
    var_37 = var_32.validate(var_36)
    var_38 = 'error at index 0'
    var_39 = 999
    var_40 = [var_39]
    var_41 = 'extra'
    var_42 = [var_7, var_41]
    var_43 = True
    var_44 = module_0.Array(unique_items=var_43)
    var_45 = 1
    var_46 = [var_45, var_45]
    var_47 = var_44.validate(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = [var_48, var_49]



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 3
    var_2 = 5
    var_3 = module_0.String(max_length=var_2, min_length=var_1)
    var_4 = 'abc'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'abc'
    var_6 = '  abc  '
    var_7 = var_3.validate(var_6)
    assert var_7 == 'abc'
    var_8 = 'ab'
    var_9 = var_3.validate(var_8)
    var_10 = 'abcdef'
    var_11 = var_3.validate(var_10)
    var_12 = True
    var_13 = module_0.String()
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = False
    var_17 = module_0.String()
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(allow_blank=var_16)
    var_21 = '   '
    var_22 = var_20.validate(var_21)
    var_23 = module_0.String(allow_blank=var_12)
    var_24 = '   '
    var_25 = var_23.validate(var_24)
    assert var_25 == ''
    var_26 = module_0.String(allow_blank=var_12, coerce_types=var_12)
    var_27 = var_26.validate(var_14)
    assert var_27 == ''
    var_28 = 123
    var_29 = var_3.validate(var_28)
    var_30 = '^\\d+$'
    var_31 = module_0.String(pattern=var_30)
    var_32 = '123'
    var_33 = var_31.validate(var_32)
    assert var_33 == '123'
    var_34 = '123a'
    var_35 = var_31.validate(var_34)
    var_36 = '^[a-z]+$'
    var_37 = module_1.compile(var_36)
    var_38 = module_0.String(pattern=var_37)
    var_39 = var_38.validate(var_4)
    assert var_39 == 'abc'
    var_40 = 'ABC'
    var_41 = var_38.validate(var_40)
    var_42 = module_0.String()
    var_43 = 'abc\x00def'
    var_44 = var_42.validate(var_43)
    assert var_44 == 'abcdef'
    var_45 = 'email'
    var_46 = module_0.String(format=var_45)
    var_47 = 'test@example.com'
    var_48 = var_46.validate(var_47)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = 'Alpha'
    var_7 = (var_1, var_6)
    var_8 = 'Beta'
    var_9 = (var_2, var_8)
    var_10 = [var_7, var_9]
    var_11 = module_0.Choice(choices=var_10)
    var_12 = (var_2, var_8)
    var_13 = [var_1, var_12]
    var_14 = module_0.Choice(choices=var_13)
    var_15 = [var_1]
    var_16 = False
    var_17 = module_0.Choice(choices=var_15, coerce_types=var_16)
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 'c'
    var_21 = (var_18, var_19, var_20)
    var_22 = [var_21]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = [var_18]
    var_25 = 'My Choice'
    var_26 = True
    var_27 = module_0.Choice(choices=var_24)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = module_0.Array(var_0, exact_items=var_3)
    var_8 = [var_2, var_3]
    var_9 = var_7.validate(var_8)
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = var_7.validate(var_13)
    var_15 = 'exact_items'
    var_16 = module_0.Array(var_0, min_items=var_11)
    var_17 = [var_10, var_11]
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Array(var_0, max_items=var_11)
    var_20 = [var_10, var_11]
    var_21 = var_19.validate(var_20)
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_19.validate(var_25)
    var_27 = 'max_items'
    var_28 = True
    var_29 = module_0.Array(var_0, unique_items=var_28)
    var_30 = [var_28, var_23, var_24]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = [var_32, var_33, var_33]
    var_35 = var_29.validate(var_34)
    var_36 = 'unique_items'
    var_37 = module_0.Integer()
    var_38 = module_0.Float()
    var_39 = [var_37, var_38]
    var_40 = module_0.Array(var_39)
    var_41 = 2.5
    var_42 = [var_28, var_41]
    var_43 = var_40.validate(var_42)
    var_44 = 1
    var_45 = 'not_a_float'
    var_46 = [var_44, var_45]
    var_47 = var_40.validate(var_46)
    var_48 = 'type'
    var_49 = module_0.Integer()
    var_50 = module_0.Float()
    var_51 = module_0.Array(var_49, var_50)
    var_52 = 3.3
    var_53 = [var_28, var_41, var_52]
    var_54 = var_51.validate(var_53)
    var_55 = 1
    var_56 = 'invalid'
    var_57 = [var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = True
    var_61 = module_0.Array(var_59)
    var_62 = None
    var_63 = [var_60, var_62, var_57]
    var_64 = var_61.validate(var_63)
    var_65 = None
    var_66 = var_61.validate(var_65)
    var_67 = 'null'
    var_68 = module_0.Integer()
    var_69 = module_0.Array(var_68)
    var_70 = 'not a list'
    var_71 = var_69.validate(var_70)



# Parsed testcases at query #3
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
    var_15 = module_0.Number(coerce_types=var_7)
    var_16 = ''
    var_17 = var_15.validate(var_16)
    assert var_17 is None



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'match'
    var_1 = 'no_match'
    var_2 = 'type'
    var_3 = 'match_second'
    var_4 = 'any_value'
    var_5 = True
    var_6 = None
    var_7 = False
    var_8 = None
    var_9 = excinfo.value.messages()[var_7]
    var_10 = var_9.code
    assert var_10 == 'null'
    var_11 = 'err'
    var_12 = 'random_string'
    var_13 = 'logic'
    var_14 = 'logic_error'
    var_15 = 'trigger_error'
    var_16 = excinfo.value.messages()[var_7]
    var_17 = var_16.code
    assert var_17 == 'logic_error'



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'success'
    var_1 = 123
    var_2 = None
    var_3 = True
    var_4 = 'val'
    var_5 = False
    var_6 = None
    var_7 = excinfo.value.messages()[var_5]
    var_8 = var_7.code
    assert var_8 == 'null'
    var_9 = 'type'
    var_10 = 'not_matching'
    var_11 = excinfo.value.messages()[var_5]
    var_12 = var_11.code
    assert var_12 == 'union'
    var_13 = 'specific'
    var_14 = 'specific_error'
    var_15 = 'trigger_specific'
    var_16 = excinfo.value.messages()[var_5]
    var_17 = var_16.code
    assert var_17 == 'specific_error'



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Text()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.Integer()
    var_6 = [var_0, var_5]
    var_7 = module_0.Union(var_6)
    var_8 = []
    var_9 = module_0.Union(var_8)
    var_10 = [var_0]
    var_11 = 'test'
    var_12 = module_0.Union(var_10)
    var_13 = module_0.Float()
    var_14 = module_0.Decimal()
    var_15 = [var_13, var_14]
    var_16 = module_0.Union(var_15)
    var_17 = var_16.any_of
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 0
    var_20 = var_16.any_of[var_19]
    var_21 = var_16.any_of[var_4]



# Parsed testcases at query #8
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
    var_7 = 'false'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = 'off'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = '1'
    var_14 = var_0.validate(var_13)
    assert var_14 is True
    var_15 = '0'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = var_0.validate(var_1)
    assert var_17 is True
    var_18 = var_0.validate(var_3)
    assert var_18 is False
    var_19 = ''
    var_20 = var_0.validate(var_19)
    assert var_20 is False
    var_21 = None
    var_22 = var_0.validate(var_21)
    var_23 = module_0.Boolean()
    var_24 = None
    var_25 = var_23.validate(var_24)
    assert var_25 is None
    var_26 = 'null'
    var_27 = var_23.validate(var_26)
    assert var_27 is None
    var_28 = 'none'
    var_29 = var_23.validate(var_28)
    assert var_29 is None
    var_30 = module_0.Boolean(coerce_types=var_3)
    var_31 = 'true'
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = var_30.validate(var_33)
    var_35 = 'maybe'
    var_36 = var_0.validate(var_35)
    var_37 = 123
    var_38 = var_0.validate(var_37)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'yellow fruit'
    var_3 = (var_1, var_2)
    var_4 = 'cherry'
    var_5 = [var_0, var_3, var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'apple'
    var_8 = module_0.Choice(choices=var_5)
    var_9 = var_8.validate(var_1)
    assert var_9 == 'banana'
    var_10 = var_6.validate(var_4)
    assert var_10 == 'cherry'
    var_11 = module_0.Choice(choices=var_5)
    var_12 = 'dragonfruit'
    var_13 = var_11.validate(var_12)
    var_14 = False
    var_15 = module_0.Choice(choices=var_5)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = True
    var_19 = module_0.Choice(choices=var_5)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = module_0.Choice(choices=var_5)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allow_null'
    var_1 = 1
    var_2 = True
    var_3 = module_0.Const(var_1)
    var_4 = 'match'
    var_5 = module_0.Const(var_4)
    var_6 = var_5.validate(var_4)
    assert var_6 == 'match'
    var_7 = 123
    var_8 = module_0.Const(var_7)
    var_9 = var_8.validate(var_7)
    assert var_9 == 123
    var_10 = 'expected'
    var_11 = module_0.Const(var_10)
    var_12 = 'actual'
    var_13 = var_11.validate(var_12)
    var_14 = None
    var_15 = module_0.Const(var_14)
    var_16 = 'not_null'
    var_17 = var_15.validate(var_16)
    var_18 = 'only_null'
    var_19 = None
    var_20 = module_0.Const(var_19)
    var_21 = var_20.validate(var_19)
    assert var_21 is None



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Const(var_0)
    var_2 = 123
    var_3 = module_0.Const(var_2)
    var_4 = None
    var_5 = module_0.Const(var_4)
    var_6 = 'test'
    var_7 = True
    var_8 = module_0.Const(var_6)
    var_9 = 'match'
    var_10 = module_0.Const(var_9)
    var_11 = var_10.validate(var_9)
    assert var_11 == 'match'
    var_12 = module_0.Const(var_9)
    var_13 = 'mismatch'
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Const(var_8)
    var_16 = 'not_none'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = 123
    var_3 = module_0.Const(var_2)
    var_4 = None
    var_5 = module_0.Const(var_4)
    var_6 = 'key'
    var_7 = 'val'
    var_8 = {var_6: var_7}
    var_9 = module_0.Const(var_8)
    var_10 = 'val'
    var_11 = True
    var_12 = module_0.Const(var_10)
    var_13 = 'match'
    var_14 = module_0.Const(var_13)
    var_15 = var_14.validate(var_13)
    assert var_15 == 'match'
    var_16 = 'mismatch'
    var_17 = var_14.validate(var_16)
    var_18 = module_0.Const(var_12)
    var_19 = var_18.validate(var_12)
    assert var_19 is None
    var_20 = 'not_none'
    var_21 = var_18.validate(var_20)



