####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_9 = 123
    var_10 = var_3.validate(var_9)
    var_11 = True
    var_12 = module_1.AllOf(var_2)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'
    var_6 = module_0.Integer()
    var_7 = module_1.IfThenElse(var_0, var_6, var_2)
    var_8 = 'hello'
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Integer()
    var_11 = module_1.IfThenElse(var_0, var_1, var_10)
    var_12 = 123
    var_13 = var_11.validate(var_12)
    assert var_13 == 123
    var_14 = module_0.String()
    var_15 = module_1.IfThenElse(var_0, var_1, var_14)
    var_16 = 123
    var_17 = var_15.validate(var_16)
    var_18 = module_0.String()
    var_19 = module_1.IfThenElse(var_18)
    var_20 = 'anything'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'anything'
    var_22 = var_19.validate(var_12)
    assert var_22 == 123



# Parsed testcases at query #3
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
    var_6 = 'test'
    var_7 = var_3.validate(var_6)
    var_8 = 123



# Parsed testcases at query #4
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
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12]
    var_14 = var_3.validate(var_13)
    var_15 = 'any'



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = module_0.Any()
    var_3 = module_0.Any()
    var_4 = module_0.Integer()
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'
    var_9 = module_0.Integer()
    var_10 = module_0.String()
    var_11 = module_0.Any()
    var_12 = module_1.IfThenElse(var_9, var_10, var_11)
    var_13 = 123
    var_14 = var_12.validate(var_13)
    assert var_14 == '123'
    var_15 = 'abc'
    var_16 = var_12.validate(var_15)
    assert var_16 == 'abc'
    var_17 = module_0.String()
    var_18 = True
    var_19 = module_1.IfThenElse(var_17)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = 'not a string'
    var_5 = var_1.validate(var_4)
    assert var_5 == 'not a string'
    var_6 = 'a string'
    var_7 = var_1.validate(var_6)
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = 'not an integer'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'not an integer'
    var_12 = 123
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #7
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
    var_6 = module_0.Any()
    var_7 = [var_6, var_6]
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



# Parsed testcases at query #8
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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
    var_7 = module_1.NeverMatch()
    var_8 = [var_7, var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = 'test'
    var_11 = var_9.validate(var_10)
    var_12 = str(var_6)
    var_13 = [var_0]
    var_14 = module_1.AllOf(var_13)
    var_15 = 'hello'
    var_16 = var_14.validate(var_15)
    assert var_16 == 'hello'



# Parsed testcases at query #11
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
    var_7 = 'test'
    var_8 = module_0.String()
    var_9 = [var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'hello'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'hello'



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_0.Integer()
    var_5 = module_1.Not(var_0)
    var_6 = 123
    var_7 = var_0.validate(var_6)
    var_8 = 'this is a string'
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #13
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    var_9 = 123
    var_10 = var_6.validate(var_9)
    assert var_10 == 123
    var_11 = '123'
    var_12 = var_6.validate(var_11)
    var_13 = True
    var_14 = module_1.IfThenElse(var_0)



# Parsed testcases at query #15
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #16
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
    var_6 = module_0.Any()
    var_7 = module_0.Any()
    var_8 = [var_6, var_7]
    var_9 = module_1.AllOf(var_8)
    var_10 = 'test'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test'
    var_12 = module_0.Any()
    var_13 = 'test'



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Integer()
    var_3 = module_1.Not(var_2)
    var_4 = True
    var_5 = module_1.Not(var_0)
    var_6 = 'test'
    var_7 = var_1.validate(var_6)
    var_8 = 'test'
    var_9 = var_1.validate(var_8)
    var_10 = var_3.validate(var_8)
    assert var_10 == 'test'



# Parsed testcases at query #18
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
    var_6 = 'abc'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'abc'
    var_8 = 123
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #19
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



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
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123
    var_9 = 'abc'
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #21
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
    var_9 = str(var_7)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'abc'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'abc'



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    assert var_3 == 123
    var_4 = 'a string'
    var_5 = var_1.validate(var_4)
    var_6 = True
    var_7 = module_1.Not(var_0)



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
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Any()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'test'
    var_15 = var_13.validate(var_14)
    var_16 = [var_0]
    var_17 = True
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #24
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
    var_8 = 'hello'
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Integer()
    var_11 = module_1.Not(var_10)
    var_12 = 'not an int'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'not an int'
    var_14 = 10
    var_15 = var_11.validate(var_14)



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
    var_5 = [var_4]
    var_6 = module_1.AllOf(var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'
    var_9 = 123
    var_10 = var_6.validate(var_9)
    var_11 = module_0.String()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)



# Parsed testcases at query #26
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
    var_6 = module_0.Any()
    var_7 = module_1.IfThenElse(var_0, var_5, var_6)
    var_8 = var_7.if_clause
    var_9 = var_7.then_clause
    var_10 = var_7.else_clause
    var_11 = 'hello'
    var_12 = var_7.validate(var_11)
    var_13 = module_0.String()
    var_14 = module_0.Any()
    var_15 = module_0.Integer()
    var_16 = module_1.IfThenElse(var_13, var_14, var_15)
    var_17 = 'hello'
    var_18 = var_16.validate(var_17)
    assert var_18 == 'hello'
    var_19 = 123
    var_20 = var_7.validate(var_19)
    assert var_20 == 123
    var_21 = module_0.String()
    var_22 = True
    var_23 = module_1.IfThenElse(var_21)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.Any()
    var_5 = [var_4, var_4]
    var_6 = module_1.AllOf(var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'
    var_9 = module_0.Any()
    var_10 = [var_0, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = 123
    var_13 = var_11.validate(var_12)
    var_14 = [var_0]
    var_15 = True
    var_16 = module_1.AllOf(var_14)



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
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = True
    var_5 = module_1.AllOf(var_2)
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = [var_6, var_7]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = var_9.all_of
    var_12 = len(var_11)
    assert var_12 == 2



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
    var_13 = 'not an int'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'not an int'
    var_15 = 10
    var_16 = var_12.validate(var_15)



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'abc'
    var_5 = var_3.validate(var_4)
    var_6 = module_0.String()
    var_7 = 123
    var_8 = True
    var_9 = module_1.AllOf(var_2)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
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
    var_4 = 123
    var_5 = var_1.validate(var_4)
    assert var_5 == 123
    var_6 = 'string_value'
    var_7 = var_1.validate(var_6)



# Parsed testcases at query #2
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'test'
    var_5 = 'test'
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #4
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
    assert var_5 == 'hello'
    var_6 = module_0.Integer()
    var_7 = module_1.Not(var_6)
    var_8 = 'not an integer'
    var_9 = var_7.validate(var_8)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'not an integer'
    var_5 = var_3.validate(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = True
    var_10 = module_1.AllOf(var_8)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'test'
    var_13 = False
    var_14 = 'test'



# Parsed testcases at query #6
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
    var_10 = module_0.String()
    var_11 = module_0.String()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'test'
    var_15 = var_13.validate(var_14)
    var_16 = [var_0]
    var_17 = True
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #7
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
    var_8 = 'hello'
    var_9 = var_5.validate(var_8)



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
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = module_0.Any()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'any'
    var_15 = var_13.validate(var_14)
    var_16 = [var_0]
    var_17 = True
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #9
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
    var_7 = 'matches'
    var_8 = var_1.validate(var_7)
    var_9 = str(var_4)



# Parsed testcases at query #10
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
    var_7 = module_1.AllOf(var_5)
    var_8 = 'test'
    var_9 = var_7.validate(var_8)
    assert var_9 == 'test'
    var_10 = 'test'



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.String()
    var_4 = [var_3]
    var_5 = module_1.AllOf(var_4)
    var_6 = 'test'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'test'
    var_8 = module_0.String()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_1.AllOf(var_10)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = var_11.validate(var_6)
    assert var_14 == 'test'
    var_15 = module_0.String()
    var_16 = [var_15]
    var_17 = True
    var_18 = module_1.AllOf(var_16)
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = [var_19, var_20]
    var_22 = module_1.AllOf(var_21)
    var_23 = 'not an integer'
    var_24 = var_22.validate(var_23)



# Parsed testcases at query #12
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()
    var_6 = None
    var_7 = var_0.validate(var_6)
    var_8 = 123
    var_9 = var_0.validate(var_8)



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = module_0.String()
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = module_1.IfThenElse(var_7, var_8, var_9)
    var_11 = 'abc'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'abc'
    var_13 = 123
    var_14 = var_10.validate(var_13)
    assert var_14 == 123
    var_15 = module_0.String()
    var_16 = True
    var_17 = module_1.IfThenElse(var_15)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'not an integer'
    var_5 = var_3.validate(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #17
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
    var_9 = 'hello'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'hello'
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_1.AllOf(var_13)
    var_15 = 'not an int'
    var_16 = var_14.validate(var_15)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Integer()
    var_3 = module_1.Not(var_2)
    var_4 = True
    var_5 = module_1.Not(var_0)
    var_6 = module_0.String()
    var_7 = module_1.Not(var_6)
    var_8 = 'some string'
    var_9 = var_7.validate(var_8)
    var_10 = str(var_9)
    var_11 = module_0.Integer()
    var_12 = module_1.Not(var_11)
    var_13 = 'not an integer'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'not an integer'



# Parsed testcases at query #19
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #20
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
    var_10 = module_0.Any()
    var_11 = [var_0, var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    var_15 = 'Matched more in one type'
    var_16 = 'multiple_matches'
    var_17 = [var_0]
    var_18 = True
    var_19 = module_1.OneOf(var_17)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause
    var_4 = module_0.Integer()
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    var_9 = 123
    var_10 = var_6.validate(var_9)
    assert var_10 == 123
    var_11 = True
    var_12 = module_1.IfThenElse(var_0)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_1.IfThenElse(var_0)
    var_5 = var_4.then_clause
    var_6 = var_4.else_clause
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = module_1.IfThenElse(var_0, var_7, var_8)
    var_10 = module_0.Integer()
    var_11 = module_0.String()
    var_12 = True
    var_13 = module_1.IfThenElse(var_0)
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_1.IfThenElse(var_14, var_15, var_16)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String()
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = module_1.IfThenElse(var_20, var_21, var_22)
    var_24 = 'test'
    var_25 = var_23.validate(var_24)
    assert var_25 == 'test'
    var_26 = module_0.Integer()
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = module_1.IfThenElse(var_26, var_27, var_28)
    var_30 = var_29.validate(var_24)
    assert var_30 == 'test'
    var_31 = 'test'
    var_32 = var_29.validate(var_31)



# Parsed testcases at query #23
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #24
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = True
    var_5 = module_0.NeverMatch()



# Parsed testcases at query #25
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)



# Parsed testcases at query #26
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
    var_6 = 'test'
    var_7 = var_3.validate(var_6)
    var_8 = 'not an integer'
    var_9 = var_3.validate(var_8)
    var_10 = module_0.String()
    var_11 = [var_10]
    var_12 = module_1.AllOf(var_11)
    var_13 = 'hello'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'hello'



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.String()
    var_4 = [var_3]
    var_5 = module_1.AllOf(var_4)
    var_6 = module_0.String()
    var_7 = [var_6]
    var_8 = 'test'
    var_9 = var_5.validate(var_8)
    assert var_9 == 'test'
    var_10 = module_0.Any()
    var_11 = module_0.Any()
    var_12 = [var_10, var_11]
    var_13 = module_1.AllOf(var_12)
    var_14 = 123
    var_15 = var_13.validate(var_14)
    assert var_15 == 123
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_1.AllOf(var_18)
    var_20 = 'not an integer'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.String()
    var_23 = [var_22]
    var_24 = True
    var_25 = module_1.AllOf(var_23)
    var_26 = 'key'
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = module_0.Any()
    var_30 = [var_29]
    var_31 = module_1.AllOf(var_30)
    var_32 = var_31.validate(var_28)



# Parsed testcases at query #28
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
    var_11 = module_0.String()
    var_12 = [var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'test'
    var_15 = var_13.validate(var_14)
    var_16 = True
    var_17 = module_1.OneOf(var_2)



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
    var_5 = module_0.Any()
    var_6 = module_1.IfThenElse(var_0, var_4, var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Any()
    var_11 = module_1.IfThenElse(var_9, var_10)
    var_12 = 'hello'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'hello'
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = module_1.IfThenElse(var_14, var_16, var_15)
    var_18 = 123
    var_19 = var_17.validate(var_18)
    assert var_19 == 123
    var_20 = 'abc'
    var_21 = var_17.validate(var_20)
    assert var_21 == 'abc'
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_1.IfThenElse(var_22)



# Parsed testcases at query #30
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
    var_6 = 'abc'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'abc'
    var_8 = 1
    var_9 = var_5.validate(var_8)



