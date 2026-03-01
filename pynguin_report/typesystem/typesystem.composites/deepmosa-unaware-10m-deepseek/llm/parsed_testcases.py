####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_1.Not(var_4)
    var_6 = 'hello'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'hello'
    var_8 = 42
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_1.Not(var_10)
    var_12 = 'not a boolean'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'not a boolean'
    var_14 = True
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = module_0.Boolean()
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Boolean()
    var_6 = module_1.IfThenElse(var_3, var_4, var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = module_0.Boolean()
    var_12 = module_1.IfThenElse(var_9, var_10, var_11)
    var_13 = 123
    var_14 = var_12.validate(var_13)
    var_15 = 0
    var_16 = module_0.Integer(minimum=var_15)
    var_17 = 100
    var_18 = module_0.Integer(maximum=var_17)
    var_19 = module_0.String()
    var_20 = module_1.IfThenElse(var_16, var_18, var_19)
    var_21 = 50
    var_22 = var_20.validate(var_21)
    assert var_22 == 50
    var_23 = module_0.Integer(minimum=var_15)
    var_24 = module_0.String()
    var_25 = module_0.Integer(maximum=var_17)
    var_26 = module_1.IfThenElse(var_23, var_24, var_25)
    var_27 = -10
    var_28 = var_26.validate(var_27)
    assert var_28 == -10
    var_29 = module_0.Integer()
    var_30 = module_0.String()
    var_31 = module_1.IfThenElse(var_29, else_clause=var_30)
    var_32 = 123
    var_33 = var_31.validate(var_32)
    assert var_33 == 123
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = module_1.IfThenElse(var_34, var_35)
    var_37 = True
    var_38 = var_36.validate(var_37)
    assert var_38 is True
    var_39 = module_0.String()
    var_40 = module_1.IfThenElse(var_39)
    var_41 = var_40.validate(var_32)
    assert var_41 == 123
    var_42 = 'test'
    var_43 = var_40.validate(var_42)
    assert var_43 == 'test'
    var_44 = var_40.validate(var_37)
    assert var_44 is True
    var_45 = 10
    var_46 = module_0.Integer(minimum=var_45)
    var_47 = module_0.String()
    var_48 = module_0.Boolean()
    var_49 = module_1.IfThenElse(var_46, var_47, var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.String()
    var_52 = module_1.IfThenElse(var_49, var_50, var_51)
    var_53 = 5
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Integer(minimum=var_15, maximum=var_17)
    var_56 = module_0.Integer(minimum=var_21)
    var_57 = module_0.Integer(maximum=var_15)
    var_58 = module_1.IfThenElse(var_55, var_56, var_57)
    var_59 = 75
    var_60 = var_58.validate(var_59)
    assert var_60 == 75
    var_61 = 25
    var_62 = var_58.validate(var_61)
    var_63 = -10
    var_64 = var_58.validate(var_63)
    assert var_64 == -10
    var_65 = 150
    var_66 = var_58.validate(var_65)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = var_3.all_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = []
    var_10 = module_1.AllOf(var_9)
    var_11 = [var_0]
    var_12 = module_1.AllOf(var_11)
    var_13 = module_0.Boolean()
    var_14 = [var_0, var_1, var_13]
    var_15 = module_1.AllOf(var_14)
    var_16 = var_15.all_of
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = var_15.all_of[var_18]
    var_20 = 1
    var_21 = var_15.all_of[var_20]
    var_22 = 2
    var_23 = var_15.all_of[var_22]
    var_24 = [var_0]
    var_25 = 'Test AllOf'
    var_26 = 'Test description'
    var_27 = module_1.AllOf(var_24)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = 123
    var_5 = var_3.validate(var_4)
    var_6 = str(var_4)
    var_7 = True
    var_8 = var_3.validate(var_7)
    assert var_8 is True
    var_9 = 3
    var_10 = module_0.String(min_length=var_9)
    var_11 = module_0.Integer()
    var_12 = module_1.IfThenElse(var_10, var_11)
    var_13 = 'hello'
    var_14 = var_12.validate(var_13)
    var_15 = 'hi'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'hi'
    var_17 = module_0.Boolean()
    var_18 = module_1.IfThenElse(var_17)
    var_19 = False
    var_20 = var_18.validate(var_19)
    assert var_20 is False
    var_21 = 'not a boolean'
    var_22 = var_18.validate(var_21)
    assert var_22 == 'not a boolean'
    var_23 = 10
    var_24 = module_0.Integer(minimum=var_23)
    var_25 = module_0.String()
    var_26 = module_0.Boolean()
    var_27 = module_1.IfThenElse(var_24, var_25, var_26)
    var_28 = 15
    var_29 = var_27.validate(var_28)
    var_30 = 5
    var_31 = var_27.validate(var_30)
    var_32 = 100
    var_33 = module_0.Integer(maximum=var_32)
    var_34 = 50
    var_35 = module_0.Integer(minimum=var_34)
    var_36 = module_0.Integer(minimum=var_19)
    var_37 = module_1.IfThenElse(var_33, var_35, var_36)
    var_38 = 75
    var_39 = var_37.validate(var_38)
    assert var_39 == 75
    var_40 = 25
    var_41 = var_37.validate(var_40)
    var_42 = -10
    var_43 = var_37.validate(var_42)
    var_44 = module_0.Integer()
    var_45 = True
    var_46 = module_1.IfThenElse(var_44)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    var_4 = str(var_3)
    var_5 = module_0.Integer()
    var_6 = module_1.Not(var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'
    var_9 = 5
    var_10 = module_0.String(max_length=var_9)
    var_11 = module_1.Not(var_10)
    var_12 = 'short'
    var_13 = var_11.validate(var_12)
    var_14 = 123
    var_15 = var_11.validate(var_14)
    assert var_15 == 123
    var_16 = 0
    var_17 = module_0.Integer(minimum=var_16)
    var_18 = module_1.Not(var_17)
    var_19 = 10
    var_20 = var_18.validate(var_19)
    var_21 = -5
    var_22 = var_18.validate(var_21)
    assert var_22 == -5
    var_23 = module_0.Integer()
    var_24 = module_1.Not(var_23)
    var_25 = None
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = module_0.Boolean()
    var_28 = module_1.Not(var_27)
    var_29 = True
    var_30 = var_28.validate(var_29)
    var_31 = 'not a boolean'
    var_32 = var_28.validate(var_31)
    assert var_32 == 'not a boolean'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 'string'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = var_1.validate(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = var_1.validate(var_6)
    assert var_7 is None
    var_8 = 42
    var_9 = var_1.validate(var_8)
    var_10 = 5
    var_11 = module_0.String(max_length=var_10)
    var_12 = module_1.Not(var_11)
    var_13 = 'hello'
    var_14 = var_12.validate(var_13)
    var_15 = 'very long string'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'very long string'
    var_17 = True
    var_18 = module_1.Not(var_0)
    var_19 = 'negated'
    var_20 = 'Custom error'
    var_21 = {var_19: var_20}
    var_22 = module_1.Not(var_0)
    var_23 = 42
    var_24 = var_22.validate(var_23)



# Parsed testcases at query #7
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_0.validate(var_3)
    var_5 = 'test'
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 42
    var_13 = var_11.validate(var_12)
    assert var_13 == 42
    var_14 = module_0.Integer()
    var_15 = module_0.Boolean()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'not a match'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Any()
    var_21 = module_0.Any()
    var_22 = [var_20, var_21]
    var_23 = module_1.OneOf(var_22)
    var_24 = 'any value'
    var_25 = var_23.validate(var_24)
    var_26 = []
    var_27 = module_1.OneOf(var_26)
    var_28 = 'anything'
    var_29 = var_27.validate(var_28)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    var_4 = 'string'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'string'
    var_6 = 3.14
    var_7 = var_1.validate(var_6)
    var_8 = True
    var_9 = var_1.validate(var_8)
    assert var_9 is True
    var_10 = module_0.String()
    var_11 = module_1.Not(var_10)
    var_12 = 'hello'
    var_13 = var_11.validate(var_12)
    var_14 = 123
    var_15 = var_11.validate(var_14)
    assert var_15 == 123
    var_16 = True
    var_17 = module_1.Not(var_0)
    var_18 = module_0.Integer()
    var_19 = module_0.Array(var_18)
    var_20 = module_1.Not(var_19)
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = var_20.validate(var_24)
    var_26 = 'not an array'
    var_27 = var_20.validate(var_26)
    assert var_27 == 'not an array'
    var_28 = 42
    var_29 = var_1.validate(var_28)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 42
    var_13 = var_11.validate(var_12)
    assert var_13 == 42
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = True
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Any()
    var_21 = module_0.Integer()
    var_22 = [var_20, var_21]
    var_23 = module_1.OneOf(var_22)
    var_24 = 42
    var_25 = var_23.validate(var_24)
    var_26 = []
    var_27 = module_1.OneOf(var_26)
    var_28 = 'anything'
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Boolean()
    var_31 = module_0.Integer()
    var_32 = module_0.String()
    var_33 = [var_31, var_32]
    var_34 = module_1.OneOf(var_33)
    var_35 = [var_30, var_34]
    var_36 = module_1.OneOf(var_35)
    var_37 = True
    var_38 = var_36.validate(var_37)
    assert var_38 is True
    var_39 = module_0.Integer()
    var_40 = [var_39]
    var_41 = module_1.OneOf(var_40)
    var_42 = 'not an integer'
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #13
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = 'test'
    var_14 = var_2.validate(var_13)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = var_3.all_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = []
    var_10 = module_1.AllOf(var_9)
    var_11 = 123
    var_12 = var_3.validate(var_11)
    assert var_12 == 123
    var_13 = 'not_a_number'
    var_14 = var_3.validate(var_13)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Integer()
    var_10 = module_0.String()
    var_11 = [var_9, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 123
    var_14 = var_12.validate(var_13)
    var_15 = 0
    var_16 = module_0.Number(minimum=var_15)
    var_17 = 10
    var_18 = module_0.Number(maximum=var_17)
    var_19 = [var_16, var_18]
    var_20 = module_1.AllOf(var_19)
    var_21 = 5
    var_22 = var_20.validate(var_21)
    assert var_22 == 5
    var_23 = 15
    var_24 = var_20.validate(var_23)
    var_25 = -5
    var_26 = var_20.validate(var_25)



# Parsed testcases at query #16
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_1.Not(var_4)
    var_6 = module_0.String()
    var_7 = module_1.Not(var_6)
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = 'not an integer'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'not an integer'
    var_12 = module_0.Any()
    var_13 = module_1.Not(var_12)
    var_14 = 'any value'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Integer()
    var_17 = 'negated'
    var_18 = 'Custom error'
    var_19 = {var_17: var_18}
    var_20 = module_1.Not(var_16)
    var_21 = 123
    var_22 = var_20.validate(var_21)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = -1
    var_18 = var_14.validate(var_17)
    var_19 = 11
    var_20 = var_14.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String(max_length=var_15)
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 42
    var_26 = var_24.validate(var_25)
    assert var_26 == 42
    var_27 = 'test'
    var_28 = var_24.validate(var_27)
    assert var_28 == 'test'
    var_29 = 'toolong'
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #19
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #20
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 123
    var_6 = var_2.validate(var_5)
    var_7 = 'test'
    var_8 = var_2.validate(var_7)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'
    var_13 = []
    var_14 = 'A'
    var_15 = 'B'
    var_16 = 'C'
    var_17 = 'test'



# Parsed testcases at query #23
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 'string'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = var_1.validate(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = var_1.validate(var_6)
    assert var_7 is None
    var_8 = 42
    var_9 = var_1.validate(var_8)
    var_10 = 5
    var_11 = module_0.String(max_length=var_10)
    var_12 = module_1.Not(var_11)
    var_13 = 'short'
    var_14 = var_12.validate(var_13)
    var_15 = 'very long string'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'very long string'
    var_17 = True
    var_18 = module_1.Not(var_0)
    var_19 = module_0.Integer()
    var_20 = 'field'
    var_21 = 42
    var_22 = {var_20: var_21}
    var_23 = 'field'
    var_24 = 'not integer'
    var_25 = {var_23: var_24}



# Parsed testcases at query #25
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = []
    var_14 = var_2.validate(var_13)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.String()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'hello'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'hello'
    var_14 = 42
    var_15 = var_11.validate(var_14)
    assert var_15 == 42
    var_16 = True
    var_17 = var_11.validate(var_16)
    var_18 = module_0.String()
    var_19 = module_0.Boolean()
    var_20 = [var_18, var_19]
    var_21 = module_1.OneOf(var_20)
    var_22 = True
    var_23 = var_21.validate(var_22)
    var_24 = 5
    var_25 = module_0.String(max_length=var_24)
    var_26 = 0
    var_27 = module_0.Integer(minimum=var_26)
    var_28 = [var_25, var_27]
    var_29 = module_1.OneOf(var_28)
    var_30 = 'test'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'test'
    var_32 = 10
    var_33 = var_29.validate(var_32)
    assert var_33 == 10
    var_34 = 'toolongstring'
    var_35 = var_29.validate(var_34)
    var_36 = -5
    var_37 = var_29.validate(var_36)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Integer()
    var_10 = 0
    var_11 = module_0.Integer(minimum=var_10)
    var_12 = [var_9, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = 5
    var_15 = var_13.validate(var_14)
    assert var_15 == 5
    var_16 = 'not an integer'
    var_17 = var_13.validate(var_16)
    var_18 = -1
    var_19 = var_13.validate(var_18)
    var_20 = []
    var_21 = module_1.AllOf(var_20)
    var_22 = 'any value'
    var_23 = var_21.validate(var_22)
    assert var_23 == 'any value'



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'any_value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'any_value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'any_value'
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Integer()
    var_13 = module_1.Not(var_12)
    var_14 = 'not_an_integer'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'not_an_integer'
    var_16 = 42
    var_17 = var_13.validate(var_16)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.OneOf(var_9)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = var_13.one_of
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 0
    var_17 = var_13.one_of[var_16]
    var_18 = 1
    var_19 = var_13.one_of[var_18]
    var_20 = 2
    var_21 = var_13.one_of[var_20]



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.OneOf(var_9)
    var_11 = [var_0]
    var_12 = 'Test Field'
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 42
    var_13 = var_11.validate(var_12)
    assert var_13 == 42
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = True
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Any()
    var_21 = module_0.Any()
    var_22 = [var_20, var_21]
    var_23 = module_1.OneOf(var_22)
    var_24 = 'anything'
    var_25 = var_23.validate(var_24)
    var_26 = []
    var_27 = module_1.OneOf(var_26)
    var_28 = 'anything'
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Integer()
    var_31 = [var_30]
    var_32 = 'Test Field'
    var_33 = module_1.OneOf(var_31)



# Parsed testcases at query #33
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'any value'
    var_6 = var_0.validate(var_5)
    var_7 = None
    var_8 = var_0.validate(var_7)
    var_9 = 123
    var_10 = var_0.validate(var_9)
    var_11 = []
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = -1
    var_18 = var_14.validate(var_17)
    var_19 = 15
    var_20 = var_14.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String(max_length=var_15)
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 42
    var_26 = var_24.validate(var_25)
    assert var_26 == 42
    var_27 = 'test'
    var_28 = var_24.validate(var_27)
    assert var_28 == 'test'
    var_29 = 'toolong'
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = var_15.validate(var_9)
    assert var_16 == 'test'
    var_17 = module_0.Any()
    var_18 = module_0.Any()
    var_19 = [var_17, var_18]
    var_20 = module_1.OneOf(var_19)
    var_21 = 'test'
    var_22 = var_20.validate(var_21)



# Parsed testcases at query #36
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'test'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'Test field'
    var_10 = module_0.NeverMatch()



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'string'
    var_13 = module_0.Field()
    var_14 = 'number'
    var_15 = module_0.Field()
    var_16 = [var_13, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    assert var_19 == 'test'
    var_20 = 123
    var_21 = var_17.validate(var_20)
    assert var_21 == 123
    var_22 = True
    var_23 = var_17.validate(var_22)
    var_24 = module_0.Any()
    var_25 = module_0.Any()
    var_26 = [var_24, var_25]
    var_27 = module_1.OneOf(var_26)
    var_28 = 'anything'
    var_29 = var_27.validate(var_28)
    var_30 = module_1.NeverMatch()
    var_31 = [var_0, var_30]
    var_32 = module_1.OneOf(var_31)
    var_33 = var_32.validate(var_18)
    assert var_33 == 'test'



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.Not(var_0)
    var_3 = module_1.NeverMatch()
    var_4 = module_1.Not(var_3)
    var_5 = 'any_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'any_value'
    var_7 = module_0.Any()
    var_8 = module_1.Not(var_7)
    var_9 = 'any_value'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Integer()
    var_12 = module_1.Not(var_11)
    var_13 = 'string'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'string'
    var_15 = 42
    var_16 = var_12.validate(var_15)
    var_17 = module_0.Any()
    var_18 = module_1.Not(var_17)



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.Not(var_1)
    var_3 = True
    var_4 = module_1.Not(var_1)
    var_5 = 'toolongstring'
    var_6 = var_2.validate(var_5)
    assert var_6 == 'toolongstring'
    var_7 = 'short'
    var_8 = var_2.validate(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = module_1.Not(var_10)
    var_12 = -5
    var_13 = var_11.validate(var_12)
    assert var_13 == -5
    var_14 = 10
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = None
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = True
    var_10 = module_1.IfThenElse(var_0)
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = module_1.IfThenElse(var_13, var_14, var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    var_19 = 3
    var_20 = module_0.String(min_length=var_19)
    var_21 = 5
    var_22 = module_0.String(max_length=var_21)
    var_23 = module_0.Integer()
    var_24 = module_1.IfThenElse(var_20, var_22, var_23)
    var_25 = 'test'
    var_26 = var_24.validate(var_25)
    assert var_26 == 'test'
    var_27 = 'ab'
    var_28 = var_24.validate(var_27)
    var_29 = 123
    var_30 = var_24.validate(var_29)
    assert var_30 == 123



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 5
    var_13 = range(var_12)
    var_14 = [Any() for _ in var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = var_15.one_of
    var_17 = len(var_16)
    assert var_17 == 5



# Parsed testcases at query #43
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = var_3.all_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = module_0.Integer()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8.all_of[var_11]
    var_13 = module_0.Boolean()
    var_14 = [var_0, var_1, var_13]
    var_15 = module_1.AllOf(var_14)
    var_16 = var_15.all_of
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = var_15.all_of[var_11]
    var_19 = 1
    var_20 = var_15.all_of[var_19]
    var_21 = 2
    var_22 = var_15.all_of[var_21]
    var_23 = [var_0]
    var_24 = True
    var_25 = module_1.AllOf(var_23)
    var_26 = []
    var_27 = module_1.AllOf(var_26)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'
    var_6 = 123
    var_7 = var_3.validate(var_6)
    assert var_7 == 123
    var_8 = True
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Boolean()
    var_11 = [var_0, var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = True
    var_14 = var_12.validate(var_13)
    var_15 = []
    var_16 = module_1.OneOf(var_15)
    var_17 = 'anything'
    var_18 = var_16.validate(var_17)
    var_19 = [var_0]
    var_20 = True
    var_21 = module_1.OneOf(var_19)



# Parsed testcases at query #46
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)
    var_11 = {}
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = 5
    var_8 = module_0.String(max_length=var_7)
    var_9 = '^[a-z]+$'
    var_10 = module_0.String(pattern=var_9)
    var_11 = [var_8, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 'abc'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'abc'
    var_15 = 'abcdef'
    var_16 = var_12.validate(var_15)
    var_17 = '123'
    var_18 = var_12.validate(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = module_0.Boolean()
    var_22 = [var_19, var_20, var_21]
    var_23 = module_1.AllOf(var_22)
    var_24 = []
    var_25 = module_1.AllOf(var_24)
    var_26 = 'any_value'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'any_value'
    var_28 = 'errors'
    var_29 = hasattr(var_25, var_28)
    var_30 = 'validate_or_error'
    var_31 = hasattr(var_25, var_30)



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.String()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'hello'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'hello'
    var_14 = 42
    var_15 = var_11.validate(var_14)
    assert var_15 == 42
    var_16 = True
    var_17 = var_11.validate(var_16)
    var_18 = module_0.Any()
    var_19 = module_0.Any()
    var_20 = [var_18, var_19]
    var_21 = module_1.OneOf(var_20)
    var_22 = 'anything'
    var_23 = var_21.validate(var_22)
    var_24 = []
    var_25 = module_1.OneOf(var_24)
    var_26 = 'anything'
    var_27 = var_25.validate(var_26)
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = module_0.Boolean()
    var_31 = [var_28, var_29, var_30]
    var_32 = module_1.OneOf(var_31)
    var_33 = True
    var_34 = var_32.validate(var_33)
    assert var_34 is True
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_1.OneOf(var_37)
    var_39 = 3.14
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Any()
    var_42 = module_0.Any()
    var_43 = [var_41, var_42]
    var_44 = module_1.OneOf(var_43)
    var_45 = 'test'
    var_46 = var_44.validate(var_45)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'hello'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'hello'
    var_11 = 123
    var_12 = var_3.validate(var_11)
    assert var_12 == 123
    var_13 = True
    var_14 = var_3.validate(var_13)
    var_15 = module_0.Boolean()
    var_16 = [var_0, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = True
    var_19 = var_17.validate(var_18)
    var_20 = [var_0, var_1, var_15]
    var_21 = module_1.OneOf(var_20)
    var_22 = var_21.one_of
    var_23 = len(var_22)
    assert var_23 == 3



# Parsed testcases at query #50
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'any_value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'any_value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'any_value'
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Any()
    var_13 = 'negated'
    var_14 = 'Custom error'
    var_15 = {var_13: var_14}
    var_16 = module_1.Not(var_12)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #52
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    var_4 = 'string'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'string'
    var_6 = True
    var_7 = var_1.validate(var_6)
    assert var_7 is True
    var_8 = None
    var_9 = var_1.validate(var_8)
    assert var_9 is None
    var_10 = 5
    var_11 = module_0.String(max_length=var_10)
    var_12 = module_1.Not(var_11)
    var_13 = 'hello'
    var_14 = var_12.validate(var_13)
    var_15 = 'too long string'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'too long string'
    var_17 = 123
    var_18 = var_12.validate(var_17)
    assert var_18 == 123
    var_19 = True
    var_20 = module_1.Not(var_0)
    var_21 = module_1.Not(var_1)
    var_22 = 42
    var_23 = var_21.validate(var_22)
    assert var_23 == 42
    var_24 = 'string'
    var_25 = var_21.validate(var_24)



# Parsed testcases at query #54
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    assert var_3 == 123
    var_4 = True
    var_5 = var_1.validate(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = var_1.validate(var_6)
    assert var_7 is None
    var_8 = 'hello'
    var_9 = var_1.validate(var_8)
    var_10 = ''
    var_11 = var_1.validate(var_10)
    var_12 = module_0.Integer()
    var_13 = module_1.Not(var_12)
    var_14 = 'string'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'string'
    var_16 = 3.14
    var_17 = var_13.validate(var_16)
    var_18 = 42
    var_19 = var_13.validate(var_18)
    var_20 = 0
    var_21 = var_13.validate(var_20)
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_1.Not(var_22)
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25)
    var_27 = module_1.Not(var_26)
    var_28 = 'single'
    var_29 = var_27.validate(var_28)
    assert var_29 == 'single'
    var_30 = var_27.validate(var_22)
    assert var_30 == 123
    var_31 = 'item1'
    var_32 = 'item2'
    var_33 = [var_31, var_32]
    var_34 = var_27.validate(var_33)



# Parsed testcases at query #56
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_0.validate(var_3)
    var_5 = 'any value'
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #57
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_0.validate(var_3)
    var_5 = 'any value'
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #58
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = 'test'
    var_14 = var_2.validate(var_13)



# Parsed testcases at query #59
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'test'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #60
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    var_13 = 3
    var_14 = 10
    var_15 = 'test'
    var_16 = var_10.validate(var_15)
    assert var_16 == 'test'
    var_17 = 'ab'
    var_18 = var_10.validate(var_17)
    var_19 = 'thisistoolong'
    var_20 = var_10.validate(var_19)
    var_21 = []
    var_22 = module_1.AllOf(var_21)
    var_23 = 'any value'
    var_24 = var_22.validate(var_23)
    assert var_24 == 'any value'
    var_25 = [var_0]
    var_26 = 'Test description'
    var_27 = module_1.AllOf(var_25)



# Parsed testcases at query #61
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.String()
    var_5 = module_0.String()
    var_6 = module_1.IfThenElse(var_4, var_5)
    var_7 = var_6.if_clause
    var_8 = var_6.then_clause
    var_9 = var_6.else_clause
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = module_0.String()
    var_13 = module_1.IfThenElse(var_10, var_11, var_12)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = module_0.Any()
    var_18 = True
    var_19 = module_1.IfThenElse(var_17)
    var_20 = 'test'
    var_21 = var_6.validate(var_20)
    var_22 = 123
    var_23 = var_6.validate(var_22)
    var_24 = 5
    var_25 = module_0.String(min_length=var_24)
    var_26 = 10
    var_27 = module_0.String(max_length=var_26)
    var_28 = module_0.Integer()
    var_29 = module_1.IfThenElse(var_25, var_27, var_28)
    var_30 = 'hello'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'hello'
    var_32 = 'hi'
    var_33 = var_29.validate(var_32)
    assert var_33 == 'hi'
    var_34 = 42
    var_35 = var_29.validate(var_34)
    assert var_35 == 42



# Parsed testcases at query #62
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = 'string'
    var_6 = module_0.Field()
    var_7 = module_1.Not(var_6)
    var_8 = 123
    var_9 = var_7.validate(var_8)
    assert var_9 == 123
    var_10 = 'hello'
    var_11 = var_7.validate(var_10)
    var_12 = module_1.Not(var_4)
    var_13 = 'anything'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'anything'
    var_15 = 'anything'



# Parsed testcases at query #63
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'any value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'any value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'any value'
    var_11 = var_9.validate(var_10)
    var_12 = 10
    var_13 = module_0.Integer(minimum=var_12)
    var_14 = module_1.Not(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = 15
    var_18 = var_14.validate(var_17)



# Parsed testcases at query #64
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Boolean()
    var_10 = [var_0, var_1, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = [var_0]
    var_15 = 'Test AllOf'
    var_16 = module_1.AllOf(var_14)



# Parsed testcases at query #65
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #66
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_1.Not(var_4)
    var_6 = 'hello'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'hello'
    var_8 = 123
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_1.Not(var_10)
    var_12 = True
    var_13 = var_11.validate(var_12)
    var_14 = False
    var_15 = var_11.validate(var_14)
    var_16 = 'string'
    var_17 = var_11.validate(var_16)
    assert var_17 == 'string'
    var_18 = 123
    var_19 = var_11.validate(var_18)
    assert var_19 == 123
    var_20 = None
    var_21 = var_11.validate(var_20)
    assert var_21 is None
    var_22 = module_0.String()
    var_23 = module_0.Array(var_22)
    var_24 = module_1.Not(var_23)
    var_25 = 'not an array'
    var_26 = var_24.validate(var_25)
    assert var_26 == 'not an array'
    var_27 = 'item1'
    var_28 = 'item2'
    var_29 = [var_27, var_28]
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #67
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = var_2.validate(var_16)



# Parsed testcases at query #68
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = True
    var_9 = var_3.validate(var_8)
    var_10 = 0
    var_11 = module_0.Integer(minimum=var_10)
    var_12 = 100
    var_13 = module_0.Integer(maximum=var_12)
    var_14 = [var_11, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 50
    var_17 = var_15.validate(var_16)
    var_18 = []
    var_19 = module_1.OneOf(var_18)
    var_20 = 'anything'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Integer()
    var_23 = [var_22]
    var_24 = True
    var_25 = module_1.OneOf(var_23)



# Parsed testcases at query #69
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'
    var_13 = []
    var_14 = 'first'
    var_15 = 'second'
    var_16 = 'third'
    var_17 = 'test'



# Parsed testcases at query #70
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'



# Parsed testcases at query #71
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = -5
    var_18 = var_14.validate(var_17)
    var_19 = 15
    var_20 = var_14.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String(max_length=var_15)
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 42
    var_26 = var_24.validate(var_25)
    assert var_26 == 42
    var_27 = 'test'
    var_28 = var_24.validate(var_27)
    assert var_28 == 'test'
    var_29 = 'toolongstring'
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #72
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.Not(var_0)
    var_3 = module_0.Any()
    var_4 = module_1.Not(var_3)
    var_5 = 'test_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test_value'
    var_7 = module_0.Any()
    var_8 = module_1.Not(var_7)
    var_9 = 'test_value'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Integer()
    var_12 = module_1.Not(var_11)
    var_13 = 'not_a_number'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'not_a_number'
    var_15 = 42
    var_16 = var_12.validate(var_15)
    var_17 = module_0.Any()
    var_18 = module_1.Not(var_17)



# Parsed testcases at query #73
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_1.AllOf(var_5)
    var_8 = 'test'
    var_9 = var_7.validate(var_8)
    var_10 = []
    var_11 = module_1.AllOf(var_10)
    var_12 = 'anything'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'anything'
    var_14 = module_0.Any()
    var_15 = module_0.Any()
    var_16 = [var_14, var_15]
    var_17 = module_1.AllOf(var_16)
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)



# Parsed testcases at query #74
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = module_0.Integer(minimum=var_9)
    var_18 = module_0.Integer(maximum=var_15)
    var_19 = [var_17, var_18]
    var_20 = module_1.AllOf(var_19)
    var_21 = 10
    var_22 = var_20.validate(var_21)
    var_23 = 1
    var_24 = module_0.String(min_length=var_23)
    var_25 = 'custom'
    var_26 = var_20.validate(var_25)
    assert var_26 == 'custom'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = module_1.OneOf(var_4)
    var_6 = []
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = var_13.one_of
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = [var_0]
    var_17 = 'Test Field'
    var_18 = 'Test description'
    var_19 = module_1.OneOf(var_16)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = 5
    var_8 = module_0.Integer(minimum=var_7)
    var_9 = 10
    var_10 = module_0.Integer(maximum=var_9)
    var_11 = [var_8, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 7
    var_14 = var_12.validate(var_13)
    assert var_14 == 7
    var_15 = 3
    var_16 = var_12.validate(var_15)
    var_17 = 12
    var_18 = var_12.validate(var_17)
    var_19 = module_0.Integer()
    var_20 = module_0.String(max_length=var_18)
    var_21 = [var_19, var_20]
    var_22 = module_1.AllOf(var_21)
    var_23 = 42
    var_24 = var_22.validate(var_23)
    var_25 = 'hello'
    var_26 = var_22.validate(var_25)
    var_27 = module_0.Integer()
    var_28 = [var_27]
    var_29 = module_1.AllOf(var_28)
    var_30 = 42
    var_31 = var_29.validate(var_30)
    assert var_31 == 42
    var_32 = []
    var_33 = module_1.AllOf(var_32)
    var_34 = 'anything'
    var_35 = var_33.validate(var_34)
    assert var_35 == 'anything'
    var_36 = module_0.Integer()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_1.AllOf(var_38)
    var_40 = 100
    var_41 = var_39.validate(var_40)



# Parsed testcases at query #3
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 'test'
    var_8 = var_0.validate(var_7)
    var_9 = 123
    var_10 = var_0.validate(var_9)
    var_11 = []
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'string'
    var_5 = module_0.Field()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = 'hello'
    var_10 = var_6.validate(var_9)
    var_11 = 'number'
    var_12 = module_0.Field()
    var_13 = module_1.Not(var_12)
    var_14 = 'text'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'text'
    var_16 = 42
    var_17 = var_13.validate(var_16)



# Parsed testcases at query #5
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #6
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 'test'
    var_8 = var_0.validate(var_7)
    var_9 = 123
    var_10 = var_0.validate(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = True
    var_10 = module_1.IfThenElse(var_0)
    var_11 = 0
    var_12 = module_0.Integer(minimum=var_11)
    var_13 = 3
    var_14 = module_0.String(min_length=var_13)
    var_15 = 2
    var_16 = module_0.String(max_length=var_15)
    var_17 = module_1.IfThenElse(var_12, var_14, var_16)
    var_18 = 5
    var_19 = var_17.validate(var_18)
    assert var_19 == '5'
    var_20 = -5
    var_21 = var_17.validate(var_20)
    assert var_21 == '-5'
    var_22 = module_0.Boolean()
    var_23 = 10
    var_24 = module_0.Integer(minimum=var_23)
    var_25 = module_0.Integer(maximum=var_11)
    var_26 = module_1.IfThenElse(var_22, var_24, var_25)
    var_27 = True
    var_28 = var_26.validate(var_27)
    var_29 = False
    var_30 = var_26.validate(var_29)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = []
    var_13 = 'Test'
    var_14 = 'Test field'
    var_15 = module_1.OneOf(var_12)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'
    var_13 = []
    var_14 = 'first'
    var_15 = 'second'
    var_16 = 'third'
    var_17 = 'test'



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_0.Integer()
    var_8 = 0
    var_9 = module_0.Integer(minimum=var_8)
    var_10 = [var_7, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = 5
    var_13 = var_11.validate(var_12)
    assert var_13 == 5
    var_14 = -1
    var_15 = var_11.validate(var_14)
    var_16 = 'not a number'
    var_17 = var_11.validate(var_16)
    var_18 = module_0.String(max_length=var_12)
    var_19 = [var_7, var_9, var_18]
    var_20 = module_1.AllOf(var_19)
    var_21 = 3
    var_22 = var_20.validate(var_21)
    var_23 = 'complex'
    var_24 = 'object'
    var_25 = {var_23: var_24}
    var_26 = module_0.Any()
    var_27 = module_0.Any()
    var_28 = [var_26, var_27]
    var_29 = module_1.AllOf(var_28)
    var_30 = var_29.validate(var_25)



# Parsed testcases at query #11
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'test'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #12
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'test'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'not an integer'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'not an integer'
    var_6 = 42
    var_7 = var_1.validate(var_6)
    var_8 = 5
    var_9 = module_0.String(min_length=var_8)
    var_10 = module_1.Not(var_9)
    var_11 = 'abc'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'abc'
    var_13 = 'abcde'
    var_14 = var_10.validate(var_13)
    var_15 = module_1.Not(var_1)
    var_16 = 42
    var_17 = var_15.validate(var_16)
    assert var_17 == 42



# Parsed testcases at query #14
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'any value'
    var_6 = var_0.validate(var_5)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.AllOf(var_9)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = var_13.all_of
    var_15 = len(var_14)
    assert var_15 == 3



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Boolean()
    var_10 = [var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 0
    var_15 = var_11.all_of[var_14]
    var_16 = 5
    var_17 = module_0.String(max_length=var_16)
    var_18 = 2
    var_19 = module_0.String(min_length=var_18)
    var_20 = [var_17, var_19]
    var_21 = module_1.AllOf(var_20)
    var_22 = 'test'
    var_23 = var_21.validate(var_22)
    assert var_23 == 'test'
    var_24 = 'toolongstring'
    var_25 = var_21.validate(var_24)
    var_26 = 'a'
    var_27 = var_21.validate(var_26)
    var_28 = '^[A-Z]+$'
    var_29 = module_0.String(pattern=var_28)
    var_30 = 3
    var_31 = module_0.String(max_length=var_30)
    var_32 = [var_29, var_31]
    var_33 = module_1.AllOf(var_32)
    var_34 = 'ABC'
    var_35 = var_33.validate(var_34)
    assert var_35 == 'ABC'
    var_36 = 'abcd'
    var_37 = var_33.validate(var_36)
    var_38 = 'abc'
    var_39 = var_33.validate(var_38)
    var_40 = module_0.Integer(minimum=var_14)
    var_41 = 100
    var_42 = module_0.Integer(maximum=var_41)
    var_43 = [var_40, var_42]
    var_44 = module_1.AllOf(var_43)
    var_45 = 50
    var_46 = var_44.validate(var_45)
    assert var_46 == 50



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_3 = module_1.Not(var_2)
    var_4 = -5
    var_5 = var_3.validate(var_4)
    assert var_5 == -5
    var_6 = 5
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_1.Not(var_2)
    var_10 = 3
    var_11 = module_0.String(min_length=var_10)
    var_12 = module_1.Not(var_11)
    var_13 = 'ab'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'ab'
    var_15 = 'abc'
    var_16 = var_12.validate(var_15)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.Not(var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    var_4 = 'hello'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'hello'
    var_6 = True
    var_7 = var_1.validate(var_6)
    assert var_7 is True
    var_8 = module_0.String()
    var_9 = module_1.Not(var_8)
    var_10 = 'test'
    var_11 = var_9.validate(var_10)
    var_12 = 123
    var_13 = var_9.validate(var_12)
    assert var_13 == 123
    var_14 = None
    var_15 = var_9.validate(var_14)
    assert var_15 is None
    var_16 = True
    var_17 = module_1.Not(var_0)
    var_18 = module_0.Boolean()
    var_19 = module_1.Not(var_18)
    var_20 = True
    var_21 = var_19.validate(var_20)
    var_22 = False
    var_23 = var_19.validate(var_22)
    var_24 = 'not boolean'
    var_25 = var_19.validate(var_24)
    assert var_25 == 'not boolean'
    var_26 = var_19.validate(var_12)
    assert var_26 == 123



# Parsed testcases at query #19
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.OneOf(var_9)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = var_13.one_of
    var_15 = len(var_14)
    assert var_15 == 3



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.Not(var_1)
    var_3 = True
    var_4 = module_1.Not(var_1)
    var_5 = 'too_long_string_for_field'
    var_6 = var_2.validate(var_5)
    assert var_6 == 'too_long_string_for_field'
    var_7 = 'short'
    var_8 = var_2.validate(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = module_1.Not(var_10)
    var_12 = -5
    var_13 = var_11.validate(var_12)
    assert var_13 == -5
    var_14 = 10
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'test_value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'test_value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Integer()
    var_13 = module_1.Not(var_12)
    var_14 = 'not a number'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'not a number'
    var_16 = 42
    var_17 = var_13.validate(var_16)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_1.Not(var_4)
    var_6 = 'hello'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'hello'
    var_8 = 42
    var_9 = var_5.validate(var_8)
    var_10 = 5
    var_11 = module_0.String(min_length=var_10)
    var_12 = module_1.Not(var_11)
    var_13 = 'hi'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'hi'
    var_15 = 'hello'
    var_16 = var_12.validate(var_15)



# Parsed testcases at query #24
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'test'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'Test Field'
    var_10 = 'A field that never validates'
    var_11 = module_0.NeverMatch()



# Parsed testcases at query #25
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 123
    var_6 = var_2.validate(var_5)
    var_7 = 'test'
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.AllOf(var_9)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = var_13.all_of
    var_15 = len(var_14)
    assert var_15 == 3



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = None
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_0.Integer()
    var_8 = 0
    var_9 = module_0.Integer(minimum=var_8)
    var_10 = [var_7, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = 5
    var_13 = var_11.validate(var_12)
    assert var_13 == 5
    var_14 = -1
    var_15 = var_11.validate(var_14)
    var_16 = 'not a number'
    var_17 = var_11.validate(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.Integer(minimum=var_17)
    var_20 = 10
    var_21 = module_0.Integer(maximum=var_20)
    var_22 = [var_18, var_19, var_21]
    var_23 = module_1.AllOf(var_22)
    var_24 = var_23.validate(var_12)
    assert var_24 == 5
    var_25 = 15
    var_26 = var_23.validate(var_25)
    var_27 = -5
    var_28 = var_23.validate(var_27)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = True
    var_10 = module_1.IfThenElse(var_0)
    var_11 = module_0.Field()
    var_12 = module_0.Field()
    var_13 = module_0.Field()
    var_14 = module_1.IfThenElse(var_11, var_12, var_13)
    var_15 = var_11.validate_or_error
    var_16 = None
    var_17 = 'test_value'
    var_18 = var_14.validate(var_17)
    assert var_18 == 'test_value'
    var_19 = 'error'
    var_20 = var_14.validate(var_17)
    assert var_20 == 'test_value'



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = module_1.IfThenElse(var_0, else_clause=var_2)
    var_10 = var_9.then_clause
    var_11 = True
    var_12 = module_1.IfThenElse(var_0)
    var_13 = 0
    var_14 = module_0.Integer(minimum=var_13)
    var_15 = 5
    var_16 = module_0.String(max_length=var_15)
    var_17 = module_0.Boolean()
    var_18 = module_1.IfThenElse(var_14, var_16, var_17)
    var_19 = 10
    var_20 = var_18.validate(var_19)
    var_21 = -5
    var_22 = var_18.validate(var_21)
    var_23 = module_0.Integer(minimum=var_13)
    var_24 = module_0.String()
    var_25 = module_0.Boolean()
    var_26 = module_1.IfThenElse(var_23, var_24, var_25)
    var_27 = var_26.validate(var_15)
    assert var_27 == '5'
    var_28 = False
    var_29 = var_26.validate(var_28)
    assert var_29 is False



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = 'Test Field'
    var_11 = module_1.OneOf(var_9)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 0
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 5
    var_16 = var_14.validate(var_15)
    assert var_16 == 5
    var_17 = -5
    var_18 = var_14.validate(var_17)
    var_19 = 15
    var_20 = var_14.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String(max_length=var_15)
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 42
    var_26 = var_24.validate(var_25)
    assert var_26 == 42
    var_27 = 'test'
    var_28 = var_24.validate(var_27)
    assert var_28 == 'test'
    var_29 = 'toolong'
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = var_3.one_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_3.one_of[var_6]
    var_8 = 1
    var_9 = var_3.one_of[var_8]
    var_10 = module_0.String()
    var_11 = [var_10]
    var_12 = True
    var_13 = module_1.OneOf(var_11)
    var_14 = []
    var_15 = module_1.OneOf(var_14)
    var_16 = 10
    var_17 = module_0.String(max_length=var_16)
    var_18 = module_0.Integer(minimum=var_6)
    var_19 = module_0.Boolean()
    var_20 = [var_17, var_18, var_19]
    var_21 = module_1.OneOf(var_20)
    var_22 = var_21.one_of
    var_23 = len(var_22)
    assert var_23 == 3
    var_24 = 2
    var_25 = var_21.one_of[var_24]
    var_26 = module_0.String()
    var_27 = [var_26]
    var_28 = 'Test Field'
    var_29 = module_1.OneOf(var_27)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Boolean()
    var_10 = [var_0, var_1, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = 'errors'
    var_15 = hasattr(var_3, var_14)
    var_16 = 'validate'
    var_17 = hasattr(var_3, var_16)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 5
    var_10 = module_0.String(max_length=var_9)
    var_11 = 2
    var_12 = module_0.String(min_length=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 'test'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'test'
    var_17 = 'toolongstring'
    var_18 = var_14.validate(var_17)
    var_19 = 'a'
    var_20 = var_14.validate(var_19)
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 'test'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = 10
    var_29 = module_0.String(max_length=var_28)
    var_30 = [var_27, var_29]
    var_31 = module_1.AllOf(var_30)
    var_32 = 'hello'
    var_33 = var_31.validate(var_32)



# Parsed testcases at query #37
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = 'test'
    var_14 = var_2.validate(var_13)



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.Not(var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = 123
    var_6 = var_2.validate(var_5)
    assert var_6 == 123
    var_7 = 0
    var_8 = module_0.Integer(minimum=var_7)
    var_9 = module_1.Not(var_8)
    var_10 = 5
    var_11 = var_9.validate(var_10)
    var_12 = -5
    var_13 = var_9.validate(var_12)
    assert var_13 == -5
    var_14 = True
    var_15 = module_1.Not(var_1)



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = module_0.Any()
    var_13 = module_1.NeverMatch()
    var_14 = module_0.Any()
    var_15 = [var_12, var_13, var_14]
    var_16 = module_1.OneOf(var_15)



# Parsed testcases at query #40
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_0.validate(var_3)
    var_5 = 'test'
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = True
    var_10 = module_1.IfThenElse(var_0)
    var_11 = module_0.Any()
    var_12 = module_0.Field()
    var_13 = module_0.Field()
    var_14 = module_1.IfThenElse(var_11, var_12, var_13)
    var_15 = []
    var_16 = var_12.validate
    var_17 = 'test_value'
    var_18 = var_14.validate(var_17)
    assert var_18 == 'test_value'
    var_19 = module_1.NeverMatch()
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = module_1.IfThenElse(var_19, var_20, var_21)
    var_23 = []
    var_24 = var_21.validate
    var_25 = var_22.validate(var_17)
    assert var_25 == 'test_value'



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Any()
    var_19 = [var_18]
    var_20 = module_1.OneOf(var_19)
    var_21 = 'nested'
    var_22 = var_20.validate(var_21)
    assert var_22 == 'nested'



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #44
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #45
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #46
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 'test'
    var_8 = var_0.validate(var_7)
    var_9 = 123
    var_10 = var_0.validate(var_9)
    var_11 = []
    var_12 = var_0.validate(var_11)
    var_13 = {}
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = True
    var_10 = module_1.IfThenElse(var_0)
    var_11 = module_0.Any()
    var_12 = module_0.Any()
    var_13 = module_1.NeverMatch()
    var_14 = module_1.IfThenElse(var_11, var_12, var_13)
    var_15 = 'test_value'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'test_value'
    var_17 = module_1.NeverMatch()
    var_18 = module_1.NeverMatch()
    var_19 = module_0.Any()
    var_20 = module_1.IfThenElse(var_17, var_18, var_19)
    var_21 = var_20.validate(var_15)
    assert var_21 == 'test_value'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = module_0.String()
    var_25 = module_1.IfThenElse(var_22, var_23, var_24)
    var_26 = 'not_an_integer'
    var_27 = var_25.validate(var_26)
    var_28 = 123
    var_29 = var_25.validate(var_28)
    assert var_29 == '123'



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = 123
    var_8 = var_3.validate(var_7)
    assert var_8 == 123
    var_9 = 'not_an_integer'
    var_10 = var_3.validate(var_9)
    var_11 = 123.45
    var_12 = var_3.validate(var_11)
    var_13 = []
    var_14 = module_1.AllOf(var_13)
    var_15 = 'any_value'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'any_value'
    var_17 = 5
    var_18 = module_0.String(max_length=var_17)
    var_19 = [var_3, var_18]
    var_20 = module_1.AllOf(var_19)
    var_21 = var_20.validate(var_12)
    assert var_21 == 123
    var_22 = 123456
    var_23 = var_20.validate(var_22)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.OneOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'test'
    var_13 = True
    var_14 = var_3.validate(var_13)
    var_15 = module_0.Any()
    var_16 = [var_15, var_0]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    var_20 = var_3.validate(var_11)
    assert var_20 == 'test'
    var_21 = 123
    var_22 = var_3.validate(var_21)
    assert var_22 == 123



# Parsed testcases at query #50
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)
    var_11 = {}
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = module_1.OneOf(var_9)
    var_11 = module_0.Boolean()
    var_12 = [var_0, var_1, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = var_13.one_of
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 0
    var_17 = var_13.one_of[var_16]
    var_18 = 1
    var_19 = var_13.one_of[var_18]
    var_20 = 2
    var_21 = var_13.one_of[var_20]



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = 'test'
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    var_19 = [var_3]
    var_20 = module_1.OneOf(var_19)
    var_21 = var_20.validate(var_9)
    assert var_21 == 'test'



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_9.validate(var_15)



# Parsed testcases at query #54
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)
    var_13 = 'test'
    var_14 = var_2.validate(var_13)



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 5
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 7
    var_16 = var_14.validate(var_15)
    assert var_16 == 7
    var_17 = 3
    var_18 = var_14.validate(var_17)
    var_19 = 12
    var_20 = var_14.validate(var_19)
    var_21 = 2
    var_22 = module_0.String(min_length=var_21)
    var_23 = module_0.String(max_length=var_9)
    var_24 = [var_22, var_23]
    var_25 = module_1.AllOf(var_24)
    var_26 = 'test'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'test'
    var_28 = 'a'
    var_29 = var_25.validate(var_28)
    var_30 = 'toolong'
    var_31 = var_25.validate(var_30)



# Parsed testcases at query #56
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.NeverMatch()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = 'test'
    var_10 = var_3.validate(var_9)
    assert var_10 == 'test'
    var_11 = None
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #57
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'any value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'any value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'any value'
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Integer()
    var_13 = module_1.Not(var_12)
    var_14 = 'string'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'string'
    var_16 = 123
    var_17 = var_13.validate(var_16)



# Parsed testcases at query #58
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 'test'
    var_8 = var_0.validate(var_7)
    var_9 = 123
    var_10 = var_0.validate(var_9)
    var_11 = []
    var_12 = var_0.validate(var_11)
    var_13 = {}
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #59
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 123
    var_6 = var_2.validate(var_5)
    var_7 = 'test'
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #60
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.Not(var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = 'hello world'
    var_6 = var_2.validate(var_5)
    assert var_6 == 'hello world'
    var_7 = 0
    var_8 = 10
    var_9 = module_0.Integer(minimum=var_7, maximum=var_8)
    var_10 = module_1.Not(var_9)
    var_11 = 5
    var_12 = var_10.validate(var_11)
    var_13 = -5
    var_14 = var_10.validate(var_13)
    assert var_14 == -5
    var_15 = 15
    var_16 = var_10.validate(var_15)
    assert var_16 == 15
    var_17 = True
    var_18 = module_1.Not(var_1)



# Parsed testcases at query #61
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = [var_4, var_5]
    var_7 = module_1.OneOf(var_6)
    var_8 = 42
    var_9 = var_7.validate(var_8)
    assert var_9 == 42
    var_10 = 'hello'
    var_11 = var_7.validate(var_10)
    assert var_11 == 'hello'
    var_12 = module_0.Integer()
    var_13 = module_0.Boolean()
    var_14 = [var_12, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'hello'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.Any()
    var_20 = [var_18, var_19]
    var_21 = module_1.OneOf(var_20)
    var_22 = 42
    var_23 = var_21.validate(var_22)
    var_24 = []
    var_25 = module_1.OneOf(var_24)
    var_26 = 'anything'
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Integer()
    var_29 = [var_28]
    var_30 = True
    var_31 = module_1.OneOf(var_29)



# Parsed testcases at query #62
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'any value'
    var_6 = var_2.validate(var_5)
    var_7 = 123
    var_8 = var_2.validate(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_2.validate(var_11)



# Parsed testcases at query #63
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Boolean()
    var_10 = [var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 0
    var_15 = var_11.all_of[var_14]
    var_16 = [var_0]
    var_17 = 'Test AllOf'
    var_18 = module_1.AllOf(var_16)



# Parsed testcases at query #64
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'
    var_6 = 123
    var_7 = var_3.validate(var_6)
    assert var_7 == 123
    var_8 = True
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = True
    var_15 = var_13.validate(var_14)
    var_16 = []
    var_17 = module_1.OneOf(var_16)
    var_18 = 'anything'
    var_19 = var_17.validate(var_18)
    var_20 = []
    var_21 = True
    var_22 = module_1.OneOf(var_20)



# Parsed testcases at query #65
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 5
    var_10 = module_0.Integer(minimum=var_9)
    var_11 = 10
    var_12 = module_0.Integer(maximum=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 7
    var_16 = var_14.validate(var_15)
    assert var_16 == 7
    var_17 = 3
    var_18 = var_14.validate(var_17)
    var_19 = 12
    var_20 = var_14.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String(max_length=var_9)
    var_23 = [var_21, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = 42
    var_26 = var_24.validate(var_25)
    assert var_26 == 42
    var_27 = 'test'
    var_28 = var_24.validate(var_27)
    assert var_28 == 'test'
    var_29 = 'toolong'
    var_30 = var_24.validate(var_29)



# Parsed testcases at query #66
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.OneOf(var_4)
    var_7 = module_1.OneOf(var_5)
    var_8 = 'test_value'
    var_9 = var_7.validate(var_8)
    assert var_9 == 'test_value'
    var_10 = 'test_value'
    var_11 = var_7.validate(var_10)
    var_12 = 'test_value'
    var_13 = var_7.validate(var_12)
    var_14 = []
    var_15 = module_1.OneOf(var_14)
    var_16 = 'anything'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #67
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.NeverMatch()
    var_5 = module_1.Not(var_4)
    var_6 = 'any value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'any value'
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = 'any value'
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Integer()
    var_13 = module_1.Not(var_12)
    var_14 = 'not a number'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'not a number'
    var_16 = 42
    var_17 = var_13.validate(var_16)



# Parsed testcases at query #68
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = [var_4, var_5]
    var_7 = module_1.OneOf(var_6)
    var_8 = 42
    var_9 = var_7.validate(var_8)
    assert var_9 == 42
    var_10 = module_0.Integer()
    var_11 = module_0.String()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = True
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Integer()
    var_17 = module_0.Any()
    var_18 = [var_16, var_17]
    var_19 = module_1.OneOf(var_18)
    var_20 = 42
    var_21 = var_19.validate(var_20)
    var_22 = []
    var_23 = module_1.OneOf(var_22)
    var_24 = 'anything'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Integer()
    var_27 = [var_26]
    var_28 = True
    var_29 = module_1.OneOf(var_27)



# Parsed testcases at query #69
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = 5
    var_10 = module_0.String(max_length=var_9)
    var_11 = '^[a-z]+$'
    var_12 = module_0.String(pattern=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 'test'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'test'
    var_17 = 'TEST'
    var_18 = var_14.validate(var_17)
    var_19 = 0
    var_20 = module_0.Integer(minimum=var_19)
    var_21 = 10
    var_22 = module_0.Integer(maximum=var_21)
    var_23 = [var_20, var_22]
    var_24 = module_1.AllOf(var_23)
    var_25 = var_24.validate(var_9)
    assert var_25 == 5
    var_26 = -1
    var_27 = var_24.validate(var_26)
    var_28 = 11
    var_29 = var_24.validate(var_28)



# Parsed testcases at query #70
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_9.validate(var_15)



# Parsed testcases at query #71
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #72
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Integer()
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.OneOf(var_5)
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 42
    var_13 = var_11.validate(var_12)
    assert var_13 == 42
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = True
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Any()
    var_21 = module_0.Any()
    var_22 = [var_20, var_21]
    var_23 = module_1.OneOf(var_22)
    var_24 = 'test'
    var_25 = var_23.validate(var_24)
    var_26 = []
    var_27 = module_1.OneOf(var_26)
    var_28 = 'anything'
    var_29 = var_27.validate(var_28)



# Parsed testcases at query #73
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'test_value'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'test_value'
    var_6 = module_0.Any()
    var_7 = module_1.Not(var_6)
    var_8 = 'any_value'
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'not a number'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'not a number'
    var_14 = 42
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #74
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #75
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = module_1.AllOf(var_6)
    var_10 = 'test_value'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test_value'
    var_12 = 'test_value'



