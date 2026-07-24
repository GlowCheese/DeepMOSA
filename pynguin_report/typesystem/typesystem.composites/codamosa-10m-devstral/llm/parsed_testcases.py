####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_1.AllOf()



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = None
    var_5 = 'then_result'
    var_6 = 'test_value'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'then_result'
    var_8 = 'error'
    var_9 = (var_4, var_8)
    var_10 = 'else_result'
    var_11 = var_3.validate(var_6)
    assert var_11 == 'else_result'
    var_12 = module_1.IfThenElse(var_0)
    var_13 = var_12.validate(var_6)
    assert var_13 == 'test_value'
    var_14 = (var_4, var_8)
    var_15 = var_12.validate(var_6)
    assert var_15 == 'test_value'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0, var_1]
    var_5 = 'Test description'
    var_6 = module_1.AllOf(var_4)
    var_7 = [var_0, var_1]
    var_8 = True
    var_9 = module_1.AllOf(var_7)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)
    var_9 = module_1.OneOf()



# Parsed testcases at query #5
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
    var_7 = True
    var_8 = module_1.IfThenElse(var_0)



# Parsed testcases at query #6
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.OneOf(var_12)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



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
    var_7 = True
    var_8 = module_1.IfThenElse(var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = module_0.Any()
    var_7 = [var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.OneOf(var_10)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = None
    var_5 = 'then_validated'
    var_6 = 'test_value'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'then_validated'
    var_8 = 'error'
    var_9 = 'else_validated'
    var_10 = var_3.validate(var_6)
    assert var_10 == 'else_validated'
    var_11 = module_1.IfThenElse(var_0)
    var_12 = var_11.validate(var_6)
    assert var_12 == 'test_value'
    var_13 = var_11.validate(var_6)
    assert var_13 == 'test_value'



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
    var_5 = 'Test description'
    var_6 = module_1.AllOf(var_4)
    var_7 = [var_0]
    var_8 = True
    var_9 = module_1.AllOf(var_7)



# Parsed testcases at query #16
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
    var_7 = module_1.AllOf()
    var_8 = 'not a list'
    var_9 = module_1.AllOf(var_8)
    var_10 = 'not a field'
    var_11 = [var_10]
    var_12 = module_1.AllOf(var_11)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = module_1.AllOf()



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Any()
    var_10 = module_0.Any()
    var_11 = [var_9, var_10]
    var_12 = module_0.Any()
    var_13 = [var_12]
    var_14 = 'Test'
    var_15 = module_1.AllOf(var_13)
    var_16 = module_0.Any()
    var_17 = [var_16]
    var_18 = module_0.Any()
    var_19 = [var_18]
    var_20 = True
    var_21 = module_1.AllOf(var_19)



# Parsed testcases at query #19
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
    var_7 = True
    var_8 = module_1.IfThenElse(var_0)



# Parsed testcases at query #20
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = 'any_value'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = module_0.Any()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_0, var_1]
    var_10 = 'Test description'
    var_11 = module_1.AllOf(var_9)
    var_12 = [var_0, var_1]
    var_13 = True
    var_14 = module_1.AllOf(var_12)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = 'any_value'
    var_6 = var_1.validate(var_5)
    var_7 = False
    var_8 = module_0.Any()
    var_9 = module_1.Not(var_8)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None



# Parsed testcases at query #25
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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
    var_7 = 'invalid'
    var_8 = var_3.validate(var_7)
    var_9 = module_0.Any()
    var_10 = [var_0, var_1, var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'value'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Any()
    var_15 = [var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'valid'
    var_18 = var_16.validate(var_17)
    assert var_18 == 'valid'



# Parsed testcases at query #28
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any value'
    var_4 = var_0.validate(var_3)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = 'Test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)



# Parsed testcases at query #34
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
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'test'
    var_13 = module_1.NeverMatch()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.AllOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_1.AllOf()



# Parsed testcases at query #36
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
    var_9 = module_1.IfThenElse(var_0, else_clause=var_2)
    var_10 = var_9.then_clause
    var_11 = True
    var_12 = module_1.IfThenElse(var_0)



# Parsed testcases at query #37
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = 'Test'
    var_3 = module_0.NeverMatch()
    var_4 = 'any value'
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.OneOf(var_6)



# Parsed testcases at query #39
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any_value'
    var_4 = var_0.validate(var_3)



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #41
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Field()
    var_4 = [var_3]
    var_5 = module_0.Field()
    var_6 = module_0.Field()
    var_7 = [var_5, var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Field()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.OneOf(var_12)
    var_15 = module_1.OneOf()



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_1.AllOf()



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)



# Parsed testcases at query #2
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = 'Test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_1.AllOf()
    var_8 = 'not a list'
    var_9 = module_1.AllOf(var_8)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = module_0.Any()
    var_7 = [var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.OneOf(var_10)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.Not()



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_1.AllOf()



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()
    var_2 = module_0.NeverMatch()
    var_3 = 'any_value'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)
    var_9 = 'test'
    var_10 = var_2.validate(var_9)
    var_11 = module_0.Any()
    var_12 = module_0.Any()
    var_13 = [var_11, var_12]
    var_14 = module_1.OneOf(var_13)
    var_15 = 'test'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Any()
    var_18 = [var_17]
    var_19 = module_1.OneOf(var_18)
    var_20 = 'test'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'test'



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = module_1.Not()



# Parsed testcases at query #15
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = module_1.AllOf(var_4)
    var_7 = module_1.AllOf()



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Field()
    var_4 = [var_3]
    var_5 = module_0.Field()
    var_6 = module_0.Field()
    var_7 = [var_5, var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Field()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.OneOf(var_10)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)
    var_9 = module_1.OneOf()
    var_10 = 'not a list'
    var_11 = module_1.OneOf(var_10)
    var_12 = 'not a field'
    var_13 = [var_12]
    var_14 = module_1.OneOf(var_13)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.OneOf(var_6)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'test'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'test'
    var_14 = module_0.Field()
    var_15 = [var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Any()
    var_20 = module_0.Any()
    var_21 = [var_19, var_20]
    var_22 = module_1.OneOf(var_21)
    var_23 = 'test'
    var_24 = var_22.validate(var_23)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Any()
    var_10 = module_0.Any()
    var_11 = [var_9, var_10]
    var_12 = module_0.Any()
    var_13 = [var_12]
    var_14 = True
    var_15 = module_1.AllOf(var_13)
    var_16 = module_1.AllOf()



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
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_7, var_8]
    var_10 = module_1.AllOf(var_9)
    var_11 = 'test'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'test'
    var_13 = module_1.NeverMatch()
    var_14 = module_0.Any()
    var_15 = [var_13, var_14]
    var_16 = module_1.AllOf(var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.AllOf(var_6)
    var_9 = module_0.Any()
    var_10 = module_0.Any()
    var_11 = [var_9, var_10]
    var_12 = module_1.AllOf(var_11)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)
    var_9 = module_1.NeverMatch()
    var_10 = [var_9]
    var_11 = module_1.OneOf(var_10)
    var_12 = 'test'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Any()
    var_15 = module_0.Any()
    var_16 = [var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Any()
    var_21 = [var_20]
    var_22 = module_1.OneOf(var_21)
    var_23 = 'test'
    var_24 = var_22.validate(var_23)
    assert var_24 == 'test'



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = [var_0]
    var_5 = 'Test'
    var_6 = module_1.AllOf(var_4)
    var_7 = [var_0]
    var_8 = True
    var_9 = module_1.AllOf(var_7)
    var_10 = []
    var_11 = module_1.AllOf(var_10)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = 'Test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.AllOf(var_6)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.Any()
    var_12 = [var_11]
    var_13 = True
    var_14 = module_1.OneOf(var_12)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = True
    var_3 = module_1.Not(var_0)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = 'Test'
    var_12 = module_1.AllOf(var_10)
    var_13 = module_0.Any()
    var_14 = [var_13]
    var_15 = module_0.Any()
    var_16 = [var_15]
    var_17 = True
    var_18 = module_1.AllOf(var_16)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = 'any_value'
    var_6 = var_1.validate(var_5)
    var_7 = module_1.NeverMatch()
    var_8 = module_1.Not(var_7)
    var_9 = 'any_value'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'any_value'



# Parsed testcases at query #33
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #34
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
    var_7 = True
    var_8 = module_1.IfThenElse(var_0)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_1.OneOf(var_6)



# Parsed testcases at query #36
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #37
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



