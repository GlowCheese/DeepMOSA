####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = 'hello'
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = 'not_an_int'
    var_5 = var_3.validate(var_4)
    var_6 = module_0.String()
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = module_1.IfThenElse(var_6, var_7, var_8)
    var_10 = 'hello'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'hello'
    var_12 = module_0.Boolean()
    var_13 = module_0.Any()
    var_14 = module_0.String()
    var_15 = module_1.IfThenElse(var_12, var_14, var_13)
    var_16 = 'not_a_bool'
    var_17 = var_15.validate(var_16)
    assert var_17 == 'not_a_bool'
    var_18 = module_0.Boolean()
    var_19 = module_0.Integer()
    var_20 = module_0.String()
    var_21 = module_1.IfThenElse(var_18, var_20, var_19)
    var_22 = 'not_a_bool'
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Integer()
    var_25 = module_1.IfThenElse(var_24)
    var_26 = 10
    var_27 = var_25.validate(var_26)
    assert var_27 == 10
    var_28 = 'string'
    var_29 = var_25.validate(var_28)
    assert var_29 == 'string'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_0]
    var_3 = module_1.AllOf(var_2)
    var_4 = 10
    var_5 = var_3.validate(var_4)
    assert var_5 == 10
    var_6 = [var_0, var_1]
    var_7 = module_1.AllOf(var_6)
    var_8 = 10
    var_9 = var_7.validate(var_8)
    var_10 = [var_0]
    var_11 = True
    var_12 = module_1.AllOf(var_10)
    var_13 = True
    var_14 = 'test'
    var_15 = False
    var_16 = 'test'



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = 'hello'
    var_4 = module_0.Integer()
    var_5 = module_1.IfThenElse(var_0, var_4, var_2)
    var_6 = 'not_an_integer'
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Integer()
    var_9 = module_0.Boolean()
    var_10 = module_0.String()
    var_11 = module_1.IfThenElse(var_8, var_10, var_9)
    var_12 = module_0.Integer()
    var_13 = module_0.String()
    var_14 = module_0.Boolean()
    var_15 = module_1.IfThenElse(var_12, var_13, var_14)
    var_16 = 'true'
    var_17 = var_15.validate(var_16)
    assert var_17 is True
    var_18 = module_0.Integer()
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = module_1.IfThenElse(var_18, var_19, var_20)
    var_22 = 'not_an_int'
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Integer()
    var_25 = module_1.IfThenElse(var_24)
    var_26 = 123
    var_27 = var_25.validate(var_26)
    assert var_27 == 123
    var_28 = 'abc'
    var_29 = var_25.validate(var_28)
    assert var_29 == 'abc'
    var_30 = module_0.Integer()
    var_31 = module_0.Integer()
    var_32 = module_0.String()
    var_33 = module_1.IfThenElse(var_30, var_31, var_32)
    var_34 = 10
    var_35 = var_33.validate(var_34)
    assert var_35 == 10
    var_36 = var_33.validate(var_28)
    assert var_36 == 'abc'
    var_37 = 10.5
    var_38 = var_33.validate(var_37)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_3)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = module_0.Integer()
    var_10 = module_1.Not(var_9)
    var_11 = 'not an int'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'not an int'
    var_13 = module_0.Any()
    var_14 = module_1.Not(var_13)
    var_15 = module_0.String()
    var_16 = module_1.Not(var_15)
    var_17 = 'any string'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Integer()
    var_20 = module_1.Not(var_19)
    var_21 = 'string'
    var_22 = var_20.validate(var_21)
    assert var_22 == 'string'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = [var_4]
    var_6 = module_1.AllOf(var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'
    var_9 = 'not a number'
    var_10 = var_3.validate(var_9)
    var_11 = module_0.String()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_0.String()
    var_16 = 'hello'



# Parsed testcases at query #7
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'test'
    var_7 = module_0.NeverMatch()



# Parsed testcases at query #8
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
    var_8 = module_0.Integer()
    var_9 = [var_8]
    var_10 = module_1.OneOf(var_9)
    var_11 = 'not an integer'
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'match both'
    var_17 = var_15.validate(var_16)
    var_18 = [var_0]
    var_19 = True
    var_20 = module_1.OneOf(var_18)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = module_0.Any()
    var_10 = module_1.IfThenElse(var_7, var_8, var_9)
    var_11 = module_0.Any()
    var_12 = module_0.Integer()
    var_13 = module_0.String()
    var_14 = module_1.IfThenElse(var_11, var_12, var_13)
    var_15 = 123
    var_16 = var_14.validate(var_15)
    assert var_16 == 123
    var_17 = module_0.String()
    var_18 = module_0.Integer()
    var_19 = module_1.IfThenElse(var_17, var_18)
    var_20 = 'abc'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_1.IfThenElse(var_22)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Any()
    var_5 = module_1.Not(var_4)
    var_6 = 'any value'
    var_7 = var_5.validate(var_6)
    var_8 = str(var_6)
    var_9 = 'any value'



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = '123'
    var_8 = var_3.validate(var_7)
    assert var_8 == '120'
    var_9 = module_0.String()
    var_10 = [var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = 'test'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'test'
    var_14 = module_0.Integer()
    var_15 = [var_14]
    var_16 = module_1.AllOf(var_15)
    var_17 = 'not_an_int'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.String()
    var_20 = [var_19]
    var_21 = True
    var_22 = module_1.AllOf(var_20)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = 'match'
    var_6 = 'no_match'
    var_7 = module_0.String()
    var_8 = [var_7]
    var_9 = True
    var_10 = module_1.AllOf(var_8)
    var_11 = module_0.String()
    var_12 = 'any'



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = True
    var_5 = module_1.AllOf(var_2)
    var_6 = module_0.String()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'test'
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 'not an integer'
    var_16 = var_14.validate(var_15)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 'test'
    var_8 = 'test'
    var_9 = module_0.String()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.AllOf(var_10)



# Parsed testcases at query #15
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'test'
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = [var_13, var_14]
    var_16 = module_1.AllOf(var_15)
    var_17 = 123
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Any()
    var_20 = [var_19]
    var_21 = True
    var_22 = module_1.AllOf(var_20)
    var_23 = module_0.Any()
    var_24 = [var_23]
    var_25 = module_1.AllOf(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42



# Parsed testcases at query #17
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_2)
    var_5 = module_0.Integer()
    var_6 = module_1.Not(var_5)
    var_7 = 'not an int'
    var_8 = module_0.String()
    var_9 = True
    var_10 = module_1.Not(var_8)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 10
    var_8 = 20
    var_9 = module_0.String()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.AllOf(var_10)
    var_13 = module_0.String()
    var_14 = [var_13]
    var_15 = module_1.AllOf(var_14)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)
    assert var_17 == 'test'



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = 123
    var_6 = var_1.validate(var_5)
    assert var_6 == 123
    var_7 = 'this is a string'
    var_8 = var_1.validate(var_7)
    var_9 = str(var_4)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'string'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'string'
    var_14 = 10
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #21
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
    var_8 = module_0.Any()
    var_9 = [var_0, var_8]
    var_10 = module_1.OneOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Integer()
    var_14 = [var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'not an integer'
    var_17 = var_15.validate(var_16)
    var_18 = [var_0]
    var_19 = True
    var_20 = module_1.OneOf(var_18)



# Parsed testcases at query #22
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
    var_8 = 'any'
    var_9 = 'any'
    var_10 = True
    var_11 = module_1.OneOf(var_2)



# Parsed testcases at query #23
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
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = 'test'
    var_11 = [var_0]
    var_12 = True
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 'a string'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_2)
    var_5 = 123
    var_6 = var_1.validate(var_5)
    assert var_6 == 123
    var_7 = module_0.String()
    var_8 = True
    var_9 = module_1.Not(var_7)



# Parsed testcases at query #25
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
    assert var_7 == 12
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Any()
    var_11 = [var_10, var_0]
    var_12 = module_1.OneOf(var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    var_15 = [var_0]
    var_16 = True
    var_17 = module_1.OneOf(var_15)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 'test'
    var_8 = module_0.String()
    var_9 = [var_8]
    var_10 = True
    var_11 = module_1.AllOf(var_9)
    var_12 = 'test'



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 123
    var_5 = var_1.validate(var_4)
    var_6 = 'hello'
    var_7 = var_1.validate(var_6)
    var_8 = str(var_7)
    var_9 = module_0.Integer()
    var_10 = module_1.Not(var_9)
    var_11 = 'not_an_int'
    var_12 = var_10.validate(var_11)



# Parsed testcases at query #28
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 1
    var_5 = var_3.validate(var_4)
    assert var_5 == 1
    var_6 = module_0.Integer()
    var_7 = [var_6]
    var_8 = True
    var_9 = module_1.AllOf(var_7)
    var_10 = module_0.Integer()
    var_11 = module_0.String()
    var_12 = [var_10, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = 'not an integer'
    var_15 = var_13.validate(var_14)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = 'a string'
    var_10 = var_6.validate(var_9)
    var_11 = module_0.Integer()
    var_12 = module_1.Not(var_11)
    var_13 = 10
    var_14 = var_12.validate(var_13)



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
    var_7 = 'hello'
    var_8 = var_3.validate(var_7)
    assert var_8 == 'hello'
    var_9 = 123
    var_10 = var_3.validate(var_9)
    assert var_10 == 123
    var_11 = module_0.Any()
    var_12 = module_0.Any()
    var_13 = [var_11, var_12]
    var_14 = module_1.OneOf(var_13)
    var_15 = 'any'
    var_16 = var_14.validate(var_15)
    var_17 = 'some value'



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = 'test'
    var_6 = module_0.Integer()
    var_7 = module_0.String()
    var_8 = [var_6, var_7]
    var_9 = module_1.AllOf(var_8)
    var_10 = 'not an integer'
    var_11 = var_9.validate(var_10)
    var_12 = True
    var_13 = module_1.AllOf(var_2)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = '123'
    var_5 = var_1.validate(var_4)
    assert var_5 == '123'
    var_6 = 'hello'
    var_7 = var_1.validate(var_6)



# Parsed testcases at query #34
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)



# Parsed testcases at query #35
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'abc'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'abc'
    var_6 = module_0.String()
    var_7 = module_1.Not(var_6)
    var_8 = 'hello'
    var_9 = var_7.validate(var_8)
    assert var_9 == 'hello'
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'not an integer'
    var_13 = var_11.validate(var_12)



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_1)
    var_7 = module_0.Any()
    var_8 = True
    var_9 = module_1.IfThenElse(var_0)
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = module_0.Any()
    var_13 = module_1.IfThenElse(var_10, var_11, var_12)
    var_14 = 'not_an_int'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Integer()
    var_17 = module_0.Integer()
    var_18 = module_0.String()
    var_19 = module_1.IfThenElse(var_16, var_17, var_18)
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == '123'



# Parsed testcases at query #38
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = 'never'
    var_4 = 'This never validates.'
    var_5 = True
    var_6 = module_0.NeverMatch()



# Parsed testcases at query #39
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #40
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #41
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
    var_7 = 'hello'
    var_8 = var_3.validate(var_7)
    assert var_8 == 'hello'
    var_9 = 123
    var_10 = var_3.validate(var_9)
    assert var_10 == 123
    var_11 = module_0.Any()
    var_12 = [var_0, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'test'
    var_15 = var_13.validate(var_14)
    var_16 = module_1.NeverMatch()
    var_17 = [var_16]
    var_18 = module_1.OneOf(var_17)
    var_19 = 'anything'
    var_20 = var_18.validate(var_19)



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = module_0.String()
    var_10 = module_1.Not(var_9)
    var_11 = 'match'
    var_12 = var_10.validate(var_11)



# Parsed testcases at query #43
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
    var_8 = 'any'
    var_9 = module_1.NeverMatch()
    var_10 = [var_9, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'any'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String()
    var_15 = [var_14]
    var_16 = True
    var_17 = module_1.OneOf(var_15)



# Parsed testcases at query #44
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
    var_8 = 'any'
    var_9 = 'any'
    var_10 = module_0.String()
    var_11 = [var_10]
    var_12 = True
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Any()
    var_2 = module_0.Integer()
    var_3 = module_0.Any()
    var_4 = [var_2, var_3]
    var_5 = module_1.AllOf(var_4)
    var_6 = 10
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'not an integer'
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Integer()
    var_14 = [var_13]
    var_15 = True
    var_16 = module_1.AllOf(var_14)
    var_17 = module_0.Integer()
    var_18 = module_0.String()
    var_19 = [var_17, var_18]
    var_20 = module_1.AllOf(var_19)



# Parsed testcases at query #46
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
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_3.validate(var_11)
    var_13 = module_0.Any()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'anything'
    var_18 = var_16.validate(var_17)
    var_19 = [var_0]
    var_20 = True
    var_21 = module_1.OneOf(var_19)
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = module_0.Any()
    var_25 = [var_22, var_23, var_24]
    var_26 = module_1.OneOf(var_25)



# Parsed testcases at query #47
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
    var_8 = module_0.Any()
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = []
    var_14 = var_12.validate(var_13)
    var_15 = 'test'
    var_16 = module_0.String()
    var_17 = [var_16]
    var_18 = True
    var_19 = module_1.OneOf(var_17)



# Parsed testcases at query #48
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
    var_8 = module_0.Any()
    var_9 = module_0.Any()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'anything'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Integer()
    var_15 = [var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'not an integer'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.String()
    var_20 = [var_19]
    var_21 = True
    var_22 = module_1.OneOf(var_20)
    var_23 = module_0.Any()
    var_24 = [var_23]
    var_25 = module_1.OneOf(var_24)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None



# Parsed testcases at query #49
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #50
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_0]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'not an int'
    var_5 = var_3.validate(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = 10
    var_10 = var_3.validate(var_9)
    assert var_10 == 10



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'
    var_9 = module_0.String()
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = module_1.IfThenElse(var_9, var_10, var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'test'
    var_15 = 123
    var_16 = var_12.validate(var_15)
    assert var_16 == 123
    var_17 = module_0.String()
    var_18 = True
    var_19 = module_1.IfThenElse(var_17)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = [var_0]
    var_11 = True
    var_12 = module_1.OneOf(var_10)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_0]
    var_3 = module_1.AllOf(var_2)
    var_4 = 10
    var_5 = var_3.validate(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_1.AllOf(var_6)
    var_8 = 10
    var_9 = var_7.validate(var_8)
    var_10 = [var_0]
    var_11 = True
    var_12 = module_1.AllOf(var_10)
    var_13 = module_0.Integer()
    var_14 = [var_13]
    var_15 = module_1.AllOf(var_14)
    var_16 = 5
    var_17 = var_15.validate(var_16)
    assert var_17 == 5



# Parsed testcases at query #3
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 'test'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_2)
    var_5 = 123
    var_6 = var_1.validate(var_5)
    assert var_6 == 123
    var_7 = module_0.String()
    var_8 = True
    var_9 = module_1.Not(var_7)



# Parsed testcases at query #5
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)
    var_6 = None
    var_7 = var_0.validate(var_6)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_2)
    var_5 = 123
    var_6 = var_1.validate(var_5)
    assert var_6 == 123
    var_7 = True
    var_8 = module_1.Not(var_0)



# Parsed testcases at query #7
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
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Any()
    var_11 = [var_0, var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    var_15 = True
    var_16 = module_1.OneOf(var_2)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = '12\\(not_a_string\\)'
    var_3 = var_1.validate(var_2)
    var_4 = 'any string'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_4)
    var_7 = module_0.Integer()
    var_8 = module_1.Not(var_7)
    var_9 = 'abc'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'abc'
    var_11 = module_0.String()
    var_12 = True
    var_13 = module_1.Not(var_11)



# Parsed testcases at query #9
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
    var_8 = module_0.Any()
    var_9 = module_0.Any()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'test'
    var_13 = var_11.validate(var_12)
    var_14 = module_1.NeverMatch()
    var_15 = [var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    var_19 = [var_0]
    var_20 = True
    var_21 = module_1.OneOf(var_19)



# Parsed testcases at query #10
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
    var_8 = module_0.Integer()
    var_9 = module_0.Boolean()
    var_10 = [var_8, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'not a number or bool'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Any()
    var_15 = module_0.String()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Integer()
    var_21 = [var_20]
    var_22 = module_1.OneOf(var_21)
    var_23 = 50
    var_24 = var_22.validate(var_23)
    assert var_24 == 50
    var_25 = 'not an int'
    var_26 = var_22.validate(var_25)



# Parsed testcases at query #11
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
    var_12 = 'match'
    var_13 = module_0.Any()
    var_14 = 'no_match'



# Parsed testcases at query #12
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
    var_8 = 'test'
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = []
    var_14 = var_12.validate(var_13)
    var_15 = module_0.String()
    var_16 = [var_15]
    var_17 = True
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.String()
    var_5 = module_1.Not(var_4)
    var_6 = 123
    var_7 = var_5.validate(var_6)
    assert var_7 == 123
    var_8 = 'not an integer'
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'a string'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'a string'
    var_14 = 42
    var_15 = var_11.validate(var_14)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 'test'
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = 123
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String()
    var_15 = [var_14]
    var_16 = True
    var_17 = module_1.AllOf(var_15)
    var_18 = 'hello'



# Parsed testcases at query #15
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
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = [var_0]
    var_11 = True
    var_12 = module_1.OneOf(var_10)
    var_13 = module_0.Any()
    var_14 = [var_0, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #16
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 'some string'
    var_8 = var_6.validate(var_7)
    var_9 = str(var_8)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'not an integer'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'not an integer'



# Parsed testcases at query #18
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
    var_8 = 'anything'
    var_9 = module_0.Any()
    var_10 = [var_0, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'test'
    var_13 = var_11.validate(var_12)
    var_14 = [var_0]
    var_15 = True
    var_16 = module_1.OneOf(var_14)



# Parsed testcases at query #19
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
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = module_0.String()
    var_11 = 'test'
    var_12 = module_0.String()
    var_13 = [var_12]
    var_14 = True
    var_15 = module_1.OneOf(var_13)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.if_clause
    var_3 = var_1.then_clause
    var_4 = var_1.else_clause
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = module_1.IfThenElse(var_0, var_5, var_6)
    var_8 = var_7.if_clause
    var_9 = var_7.then_clause
    var_10 = var_7.else_clause
    var_11 = '123'
    var_12 = var_7.validate(var_11)
    var_13 = 123
    var_14 = var_7.validate(var_13)
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = module_1.IfThenElse(var_15, var_16, var_17)
    var_19 = 'hello'
    var_20 = var_18.validate(var_19)
    assert var_20 == 'hello'
    var_21 = module_0.String()
    var_22 = True
    var_23 = module_1.IfThenElse(var_21)



# Parsed testcases at query #22
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
    var_8 = []
    var_9 = var_3.validate(var_8)
    var_10 = module_0.String()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'test'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.String()
    var_17 = [var_16]
    var_18 = True
    var_19 = module_1.OneOf(var_17)



# Parsed testcases at query #23
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'
    var_6 = 123
    var_7 = var_3.validate(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.AllOf(var_8)
    var_11 = module_0.Integer()
    var_12 = [var_11, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = 10
    var_15 = var_13.validate(var_14)
    assert var_15 == 10



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = '1'
    var_8 = var_3.validate(var_7)
    assert var_8 == '1'
    var_9 = []
    var_10 = var_3.validate(var_9)
    var_11 = module_0.String()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = 10
    var_16 = 20
    var_17 = 10



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 'test'
    var_8 = 'test'
    var_9 = module_0.String()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.AllOf(var_10)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    assert var_3 == 123
    var_4 = 'hello'
    var_5 = var_1.validate(var_4)
    var_6 = module_0.String()
    var_7 = True
    var_8 = module_1.Not(var_6)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = module_1.Not(var_12)
    var_14 = 'any_value'
    var_15 = var_13.validate(var_14)
    assert var_15 == 'any_value'



# Parsed testcases at query #28
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = True
    var_8 = module_1.IfThenElse(var_0)
    var_9 = module_0.String()
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = module_1.IfThenElse(var_9, var_10, var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'test'
    var_15 = 123
    var_16 = var_12.validate(var_15)
    var_17 = 123
    var_18 = var_12.validate(var_17)
    assert var_18 == 123
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = module_0.Any()
    var_22 = module_1.IfThenElse(var_19, var_20, var_21)
    var_23 = 'string_input'
    var_24 = var_22.validate(var_23)
    var_25 = module_0.String()
    var_26 = module_0.Any()
    var_27 = module_0.Integer()
    var_28 = module_1.IfThenElse(var_25, var_26, var_27)
    var_29 = 123
    var_30 = var_28.validate(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Any()
    var_33 = module_0.String()
    var_34 = module_1.IfThenElse(var_31, var_32, var_33)
    var_35 = module_0.String()
    var_36 = module_0.Any()
    var_37 = module_0.Boolean()
    var_38 = module_1.IfThenElse(var_35, var_36, var_37)
    var_39 = 123
    var_40 = var_38.validate(var_39)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 123
    var_5 = var_1.validate(var_4)
    assert var_5 == 123
    var_6 = 'hello'
    var_7 = var_1.validate(var_6)
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = 123
    var_11 = var_9.validate(var_10)
    var_12 = 'abc'
    var_13 = var_9.validate(var_12)
    assert var_13 == 'abc'



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = 'test'
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_1.AllOf(var_4)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Any()
    var_3 = [var_0, var_2]
    var_4 = module_1.AllOf(var_3)
    var_5 = module_0.Any()
    var_6 = [var_0, var_5]
    var_7 = 'test'
    var_8 = var_4.validate(var_7)
    assert var_8 == 'test'
    var_9 = module_0.Integer()
    var_10 = module_0.Any()
    var_11 = [var_9, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 'not an int'
    var_14 = var_12.validate(var_13)
    var_15 = module_0.String()
    var_16 = [var_15]
    var_17 = True
    var_18 = module_1.AllOf(var_16)
    var_19 = module_0.Any()
    var_20 = module_0.Any()
    var_21 = [var_19, var_20]
    var_22 = module_1.AllOf(var_21)
    var_23 = 123
    var_24 = var_22.validate(var_23)
    assert var_24 == 123



# Parsed testcases at query #33
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    assert var_3 == 'some_value'
    var_4 = 'some_value'
    var_5 = 'some_value'



# Parsed testcases at query #35
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = True
    var_4 = module_0.NeverMatch()



# Parsed testcases at query #36
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Any()
    var_5 = module_1.Not(var_4)
    var_6 = 'any value'
    var_7 = var_5.validate(var_6)
    var_8 = str(var_7)
    var_9 = 'a string'



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'hello'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_4)
    var_7 = 123
    var_8 = var_1.validate(var_7)
    assert var_8 == 123



# Parsed testcases at query #39
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = 'hello'
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #41
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
    var_8 = module_0.String()
    var_9 = 'test'
    var_10 = module_0.Integer()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = module_1.NeverMatch()
    var_15 = module_1.NeverMatch()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'anything'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String()
    var_21 = [var_20]
    var_22 = True
    var_23 = module_1.OneOf(var_21)



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'
    var_6 = module_0.String()
    var_7 = 123
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_0.String()
    var_11 = [var_10]
    var_12 = True
    var_13 = module_1.AllOf(var_11)



# Parsed testcases at query #43
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = 'string'
    var_8 = var_3.validate(var_7)
    var_9 = module_0.Any()
    var_10 = module_0.Any()
    var_11 = [var_9, var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 123
    var_14 = var_12.validate(var_13)
    assert var_14 == 123
    var_15 = 'test'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'test'
    var_17 = module_0.String()
    var_18 = [var_17]
    var_19 = True
    var_20 = module_1.AllOf(var_18)
    var_21 = module_0.Any()
    var_22 = [var_21]
    var_23 = module_1.AllOf(var_22)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 123
    var_12 = var_10.validate(var_11)
    assert var_12 == 123
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = [var_13, var_14]
    var_16 = module_1.AllOf(var_15)
    var_17 = 123
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Any()
    var_20 = [var_19]
    var_21 = True
    var_22 = module_1.AllOf(var_20)
    var_23 = module_0.Any()
    var_24 = [var_23]
    var_25 = module_1.AllOf(var_24)
    var_26 = 'test'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'test'



# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Any()
    var_5 = module_1.Not(var_4)
    var_6 = 'test'
    var_7 = var_5.validate(var_6)
    var_8 = str(var_6)
    var_9 = 123



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    assert var_3 == 'some_value'
    var_4 = 'some_value'
    var_5 = module_0.Any()
    var_6 = module_1.Not(var_5)
    var_7 = 'some_value'
    var_8 = var_6.validate(var_7)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 123
    var_5 = var_1.validate(var_4)
    assert var_5 == 123
    var_6 = 'hello'
    var_7 = var_1.validate(var_6)
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = 10
    var_11 = var_9.validate(var_10)
    var_12 = 'not an int'
    var_13 = var_9.validate(var_12)
    assert var_13 == 'not an int'



# Parsed testcases at query #50
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.String()
    var_5 = [var_4]
    var_6 = module_1.AllOf(var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'
    var_9 = True
    var_10 = module_1.AllOf(var_2)
    var_11 = 'not an integer'
    var_12 = var_3.validate(var_11)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.String()
    var_5 = module_1.Not(var_4)
    var_6 = 123
    var_7 = var_5.validate(var_6)
    assert var_7 == 123
    var_8 = 'some string'
    var_9 = var_5.validate(var_8)



