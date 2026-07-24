####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
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
    var_4 = var_3.all_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0, var_1]
    var_16 = 'Test'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test IfThenElse.validate() method'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = module_0.String()
    var_4 = module_1.IfThenElse(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test'
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = module_0.Boolean()
    var_10 = module_1.IfThenElse(var_7, var_8, var_9)
    var_11 = True
    var_12 = var_10.validate(var_11)
    assert var_12 is True
    var_13 = module_0.String()
    var_14 = None
    var_15 = module_0.Integer()
    var_16 = module_1.IfThenElse(var_13, var_14, var_15)
    var_17 = var_16.validate(var_5)
    assert var_17 == 'test'
    var_18 = module_0.Integer()
    var_19 = module_0.String()
    var_20 = module_1.IfThenElse(var_18, var_19, var_14)
    var_21 = var_20.validate(var_5)
    assert var_21 == 'test'
    var_22 = module_0.String()
    var_23 = module_1.IfThenElse(var_22, var_14, var_14)
    var_24 = var_23.validate(var_5)
    assert var_24 == 'test'
    var_25 = module_0.String()
    var_26 = module_0.Integer()
    var_27 = module_0.String()
    var_28 = module_1.IfThenElse(var_25, var_26, var_27)
    var_29 = 42
    var_30 = var_28.validate(var_29)
    assert var_30 == 42
    var_31 = module_0.Integer()
    var_32 = module_0.Integer()
    var_33 = module_0.String()
    var_34 = module_1.IfThenElse(var_31, var_32, var_33)
    var_35 = var_34.validate(var_5)
    assert var_35 == 'test'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = module_0.Field()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_0]
    var_10 = True
    var_11 = module_1.AllOf(var_9)
    var_12 = [var_0, var_1]
    var_13 = None
    var_14 = module_1.AllOf(var_12)
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = [var_0, var_1, var_15, var_16, var_17]
    var_19 = module_1.AllOf(var_18)
    var_20 = var_19.all_of
    var_21 = len(var_20)
    assert var_21 == 5



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = [var_0]
    var_12 = False
    var_13 = module_1.OneOf(var_11)
    var_14 = [var_0, var_1]
    var_15 = 'test description'
    var_16 = module_1.OneOf(var_14)
    var_17 = module_1.OneOf()



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
    var_7 = module_1.IfThenElse(var_0, var_1)
    var_8 = var_7.else_clause
    var_9 = module_1.IfThenElse(var_0, else_clause=var_2)
    var_10 = var_9.then_clause
    var_11 = True
    var_12 = module_1.IfThenElse(var_0)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.Any()
    var_6 = 'Test description'
    var_7 = module_1.Not(var_5)
    var_8 = module_1.Not()



# Parsed testcases at query #7
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #8
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = module_0.String()
    var_18 = [var_15, var_16, var_17]
    var_19 = module_1.AllOf(var_18)
    var_20 = var_19.all_of
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = [var_0]
    var_23 = 'test description'
    var_24 = module_1.AllOf(var_22)



# Parsed testcases at query #9
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
    var_11 = [var_0, var_1]
    var_12 = 'Test field'
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #10
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test AllOf field constructor.'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_1.AllOf(var_3)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_1]
    var_10 = module_1.AllOf(var_9)
    var_11 = var_10.all_of
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = [var_1]
    var_14 = True
    var_15 = module_1.AllOf(var_13)
    var_16 = [var_1]
    var_17 = 'Test field'
    var_18 = module_1.AllOf(var_16)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = 'test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)
    var_6 = False
    var_7 = module_1.Not(var_0)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = [var_0]
    var_5 = module_1.OneOf(var_4)
    var_6 = []
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = [var_0]
    var_12 = False
    var_13 = module_1.OneOf(var_11)
    var_14 = [var_0, var_1]
    var_15 = 'Test field'
    var_16 = module_1.OneOf(var_14)



# Parsed testcases at query #14
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
    var_9 = [var_0]
    var_10 = True
    var_11 = module_1.OneOf(var_9)
    var_12 = [var_0, var_1]
    var_13 = 'test description'
    var_14 = module_1.OneOf(var_12)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = var_6.negated
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = var_9.negated
    var_11 = module_0.Field()
    var_12 = 'test description'
    var_13 = module_1.Not(var_11)



# Parsed testcases at query #16
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'anything'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)
    var_13 = []
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.AllOf(var_8)
    var_11 = [var_0, var_1]
    var_12 = 'test'
    var_13 = module_1.AllOf(var_11)



# Parsed testcases at query #18
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 42
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #19
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0]
    var_16 = 'test description'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #20
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()



# Parsed testcases at query #21
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0, var_1]
    var_16 = None
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = [var_0, var_1]
    var_12 = 'test'
    var_13 = module_1.OneOf(var_11)
    var_14 = module_0.Field()
    var_15 = module_0.Field()
    var_16 = [var_0, var_1, var_14, var_15]
    var_17 = module_1.OneOf(var_16)
    var_18 = var_17.one_of
    var_19 = len(var_18)
    assert var_19 == 4



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = var_6.negated
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = var_9.negated
    var_11 = module_0.Any()
    var_12 = 'test description'
    var_13 = module_1.Not(var_11)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.Field()
    var_6 = 'test description'
    var_7 = module_1.Not(var_5)
    var_8 = module_1.Not()



# Parsed testcases at query #25
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
    var_11 = [var_0]
    var_12 = False
    var_13 = module_1.OneOf(var_11)
    var_14 = [var_0, var_1]
    var_15 = 'test'
    var_16 = module_1.OneOf(var_14)



# Parsed testcases at query #26
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any_value'
    var_4 = var_0.validate(var_3)
    var_5 = str(var_4)
    var_6 = None
    var_7 = var_0.validate(var_6)
    var_8 = 123
    var_9 = var_0.validate(var_8)
    var_10 = {}
    var_11 = var_0.validate(var_10)
    var_12 = []
    var_13 = var_0.validate(var_12)
    var_14 = 'test field'
    var_15 = module_0.NeverMatch()



# Parsed testcases at query #27
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
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_0, var_1, var_6, var_7, var_8]
    var_10 = module_1.OneOf(var_9)
    var_11 = var_10.one_of
    var_12 = len(var_11)
    assert var_12 == 5
    var_13 = [var_0]
    var_14 = True
    var_15 = module_1.OneOf(var_13)
    var_16 = [var_0]
    var_17 = 'test description'
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test IfThenElse constructor.'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = module_0.String()
    var_4 = module_1.IfThenElse(var_1, var_2, var_3)
    var_5 = module_1.IfThenElse(var_1)
    var_6 = var_5.then_clause
    var_7 = var_5.else_clause
    var_8 = module_1.IfThenElse(var_1, var_2)
    var_9 = var_8.else_clause
    var_10 = module_1.IfThenElse(var_1, else_clause=var_3)
    var_11 = var_10.then_clause
    var_12 = True
    var_13 = module_1.IfThenElse(var_1)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.Field()
    var_6 = 'Test not field'
    var_7 = module_1.Not(var_5)
    var_8 = module_1.Not()



# Parsed testcases at query #30
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #31
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
    var_11 = [var_0]
    var_12 = False
    var_13 = module_1.OneOf(var_11)
    var_14 = [var_0, var_1]
    var_15 = module_1.OneOf(var_14)
    var_16 = module_1.OneOf()



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = 'test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)
    var_6 = False
    var_7 = module_1.Not(var_0)
    var_8 = 'errors'
    var_9 = hasattr(var_1, var_8)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.Any()
    var_6 = 'Test not field'
    var_7 = module_1.Not(var_5)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = False
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)



# Parsed testcases at query #35
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = module_0.String()
    var_16 = [var_0, var_1, var_15]
    var_17 = module_1.AllOf(var_16)
    var_18 = var_17.all_of
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = [var_0]
    var_21 = 'Test description'
    var_22 = module_1.AllOf(var_20)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = var_3.one_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = module_0.Any()
    var_7 = [var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = []
    var_12 = module_1.OneOf(var_11)
    var_13 = var_12.one_of
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = module_0.Any()
    var_16 = [var_15]
    var_17 = True
    var_18 = module_1.OneOf(var_16)
    var_19 = module_0.Any()
    var_20 = [var_19]
    var_21 = 'test description'
    var_22 = module_1.OneOf(var_20)
    var_23 = 'description'
    var_24 = hasattr(var_22, var_23)



# Parsed testcases at query #37
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0]
    var_16 = 'test description'
    var_17 = module_1.AllOf(var_15)
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = module_0.String()
    var_21 = [var_18, var_19, var_20]
    var_22 = module_1.AllOf(var_21)
    var_23 = var_22.all_of
    var_24 = len(var_23)
    assert var_24 == 3



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Field()
    var_5 = [var_4]
    var_6 = module_1.OneOf(var_5)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = [var_0]
    var_10 = True
    var_11 = module_1.OneOf(var_9)
    var_12 = module_0.Field()
    var_13 = module_0.Field()
    var_14 = module_0.Field()
    var_15 = [var_0, var_1, var_12, var_13, var_14]
    var_16 = module_1.OneOf(var_15)
    var_17 = var_16.one_of
    var_18 = len(var_17)
    assert var_18 == 5
    var_19 = [var_0]
    var_20 = 'test description'
    var_21 = module_1.OneOf(var_19)



# Parsed testcases at query #39
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'test'
    var_8 = 'Test Field'
    var_9 = module_0.NeverMatch()



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = [var_4]
    var_6 = module_1.OneOf(var_5)
    var_7 = []
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Any()
    var_10 = [var_9]
    var_11 = True
    var_12 = module_1.OneOf(var_10)
    var_13 = module_0.Any()
    var_14 = [var_13]
    var_15 = 'test field'
    var_16 = module_1.OneOf(var_14)
    var_17 = module_0.Any()
    var_18 = [var_17]



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = var_6.negated
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = var_9.negated
    var_11 = module_0.Field()
    var_12 = 'test description'
    var_13 = module_1.Not(var_11)
    var_14 = module_0.String()
    var_15 = module_1.Not(var_14)
    var_16 = module_0.Integer()
    var_17 = module_1.Not(var_16)
    var_18 = var_15.negated
    var_19 = var_17.negated



# Parsed testcases at query #2
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)
    var_13 = []
    var_14 = var_0.validate(var_13)
    var_15 = {}
    var_16 = var_0.validate(var_15)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test AllOf field constructor.'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_1.AllOf(var_3)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = module_0.String()
    var_8 = [var_7]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = []
    var_13 = module_1.AllOf(var_12)
    var_14 = var_13.all_of
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = [var_1]
    var_17 = True
    var_18 = module_1.AllOf(var_16)
    var_19 = [var_1]
    var_20 = 'Test description'
    var_21 = module_1.AllOf(var_19)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = var_6.negated
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = var_9.negated
    var_11 = module_0.Field()
    var_12 = 'test description'
    var_13 = module_1.Not(var_11)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test AllOf constructor'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_1.AllOf(var_3)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_1]
    var_10 = module_1.AllOf(var_9)
    var_11 = var_10.all_of
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = [var_1]
    var_14 = True
    var_15 = module_1.AllOf(var_13)
    var_16 = module_0.String()
    var_17 = [var_1, var_2, var_16]
    var_18 = module_1.AllOf(var_17)
    var_19 = var_18.all_of
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = [var_1]
    var_22 = False
    var_23 = module_1.AllOf(var_21)



# Parsed testcases at query #6
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'anything'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)
    var_13 = {}
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #7
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
    var_11 = [var_0, var_1]
    var_12 = 'test description'
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #8
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
    var_9 = [var_0]
    var_10 = True
    var_11 = module_1.OneOf(var_9)
    var_12 = [var_0, var_1]
    var_13 = module_1.OneOf(var_12)
    var_14 = 'one_of'
    var_15 = hasattr(var_13, var_14)



# Parsed testcases at query #9
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
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = module_0.Any()
    var_14 = [var_11, var_12, var_13]
    var_15 = module_1.OneOf(var_14)
    var_16 = var_15.one_of
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = [var_0]
    var_19 = 'test description'
    var_20 = module_1.OneOf(var_18)



# Parsed testcases at query #10
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #11
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test description'
    var_6 = module_0.NeverMatch()
    var_7 = 'any value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = [var_0, var_1]
    var_12 = 'test description'
    var_13 = module_1.OneOf(var_11)
    var_14 = [var_0]
    var_15 = False
    var_16 = module_1.OneOf(var_14)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = module_0.Field()
    var_7 = [var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = 5
    var_10 = range(var_9)
    var_11 = [Field() for _ in var_10]
    var_12 = module_1.OneOf(var_11)
    var_13 = module_0.Field()
    var_14 = [var_13]
    var_15 = True
    var_16 = module_1.OneOf(var_14)
    var_17 = module_0.Field()
    var_18 = [var_17]
    var_19 = 'test'
    var_20 = module_1.OneOf(var_18)
    var_21 = module_0.Field()
    var_22 = module_0.Field()
    var_23 = module_0.Field()
    var_24 = [var_21, var_22, var_23]
    var_25 = module_1.OneOf(var_24)
    var_26 = var_25.one_of
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = var_25.one_of



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = 'Test description'
    var_3 = module_1.Not(var_0)
    var_4 = True
    var_5 = module_1.Not(var_0)
    var_6 = False
    var_7 = module_1.Not(var_0)
    var_8 = module_0.Field()
    var_9 = module_1.Not(var_8)



# Parsed testcases at query #15
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'any_value'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = var_3.one_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = []
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.OneOf(var_8)
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.OneOf(var_12)
    var_15 = [var_0, var_1]
    var_16 = 'test'
    var_17 = module_1.OneOf(var_15)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test OneOf field constructor.'
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_1, var_2]
    var_4 = module_1.OneOf(var_3)
    var_5 = var_4.one_of
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = [var_1]
    var_8 = module_1.OneOf(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = []
    var_12 = module_1.OneOf(var_11)
    var_13 = [var_1]
    var_14 = True
    var_15 = module_1.OneOf(var_13)
    var_16 = [var_1, var_2]
    var_17 = 'test'
    var_18 = module_1.OneOf(var_16)



# Parsed testcases at query #18
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
    var_7 = module_0.Any()
    var_8 = module_0.Any()
    var_9 = [var_0, var_1, var_6, var_7, var_8]
    var_10 = module_1.OneOf(var_9)
    var_11 = var_10.one_of
    var_12 = len(var_11)
    assert var_12 == 5
    var_13 = [var_0]
    var_14 = True
    var_15 = module_1.OneOf(var_13)
    var_16 = [var_0]
    var_17 = 'test description'
    var_18 = module_1.OneOf(var_16)
    var_19 = [var_0]
    var_20 = module_1.OneOf(var_19)
    var_21 = var_20.one_of
    var_22 = len(var_21)
    assert var_22 == 1



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.OneOf(var_6)
    var_8 = [var_0, var_1]
    var_9 = True
    var_10 = module_1.OneOf(var_8)
    var_11 = [var_0, var_1]
    var_12 = 'test description'
    var_13 = module_1.OneOf(var_11)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_0.Field()
    var_5 = [var_4]
    var_6 = module_1.AllOf(var_5)
    var_7 = []
    var_8 = module_1.AllOf(var_7)
    var_9 = [var_0]
    var_10 = True
    var_11 = module_1.AllOf(var_9)
    var_12 = [var_0, var_1]
    var_13 = module_1.AllOf(var_12)
    var_14 = 5
    var_15 = range(var_14)
    var_16 = [Field() for _ in var_15]
    var_17 = module_1.AllOf(var_16)
    var_18 = var_17.all_of
    var_19 = len(var_18)
    assert var_19 == 5



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
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
    var_13 = 'test description'
    var_14 = module_1.IfThenElse(var_0)
    var_15 = var_14.then_clause
    var_16 = var_14.else_clause



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = []
    var_5 = module_1.OneOf(var_4)
    var_6 = module_0.Field()
    var_7 = [var_6]
    var_8 = module_1.OneOf(var_7)
    var_9 = module_0.Field()
    var_10 = module_0.Field()
    var_11 = module_0.Field()
    var_12 = [var_9, var_10, var_11]
    var_13 = module_1.OneOf(var_12)
    var_14 = module_0.Field()
    var_15 = [var_14]
    var_16 = True
    var_17 = module_1.OneOf(var_15)
    var_18 = module_0.Field()
    var_19 = [var_18]
    var_20 = False
    var_21 = module_1.OneOf(var_19)
    var_22 = module_0.Field()
    var_23 = [var_22]
    var_24 = 'test'
    var_25 = module_1.OneOf(var_23)
    var_26 = module_0.Field()
    var_27 = [var_26]



# Parsed testcases at query #23
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0, var_1]
    var_16 = 'test'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #24
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = False
    var_4 = module_0.NeverMatch()
    var_5 = 'test field'
    var_6 = module_0.NeverMatch()
    var_7 = 'anything'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = var_0.validate(var_9)
    var_11 = 123
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #25
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = True
    var_2 = module_0.NeverMatch()
    var_3 = 'any_value'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = []
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #26
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
    var_6 = module_0.String()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = []
    var_12 = module_1.AllOf(var_11)
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = [var_0]
    var_16 = True
    var_17 = module_1.AllOf(var_15)
    var_18 = [var_0]
    var_19 = 'Test description'
    var_20 = module_1.AllOf(var_18)



# Parsed testcases at query #27
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0, var_1]
    var_16 = 'test description'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #28
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0]
    var_16 = 'test description'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #29
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
    var_6 = []
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = module_1.AllOf(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = [var_0]
    var_13 = True
    var_14 = module_1.AllOf(var_12)
    var_15 = [var_0]
    var_16 = 'test description'
    var_17 = module_1.AllOf(var_15)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Field()
    var_3 = True
    var_4 = module_1.Not(var_2)
    var_5 = module_0.String()
    var_6 = module_1.Not(var_5)
    var_7 = var_6.negated
    var_8 = module_0.Integer()
    var_9 = module_1.Not(var_8)
    var_10 = var_9.negated
    var_11 = module_0.Field()
    var_12 = 'test description'
    var_13 = module_1.Not(var_11)



# Parsed testcases at query #31
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
    var_6 = module_0.String()
    var_7 = [var_6]
    var_8 = module_1.AllOf(var_7)
    var_9 = []
    var_10 = module_1.AllOf(var_9)
    var_11 = [var_0]
    var_12 = True
    var_13 = module_1.AllOf(var_11)
    var_14 = [var_0, var_1]
    var_15 = 'Test field'
    var_16 = module_1.AllOf(var_14)



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
    var_5 = module_0.Integer()
    var_6 = module_1.Not(var_5)
    var_7 = module_0.String()
    var_8 = module_1.Not(var_7)
    var_9 = module_0.Any()
    var_10 = module_1.Not(var_9)



# Parsed testcases at query #33
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
    var_13 = 'test description'
    var_14 = module_1.IfThenElse(var_0)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = []
    var_5 = module_1.AllOf(var_4)
    var_6 = [var_0]
    var_7 = module_1.AllOf(var_6)
    var_8 = [var_0]
    var_9 = True
    var_10 = module_1.AllOf(var_8)
    var_11 = [var_0, var_1]
    var_12 = 'Test field'
    var_13 = module_1.AllOf(var_11)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
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
    var_13 = 'test'
    var_14 = module_1.IfThenElse(var_0)



