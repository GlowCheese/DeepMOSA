####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = 1
    var_5 = var_3.validate(var_4)
    assert var_5 == 1
    var_6 = module_1.NeverMatch()
    var_7 = module_0.Field()
    var_8 = module_0.Field()
    var_9 = module_1.IfThenElse(var_6, var_7, var_8)
    var_10 = var_9.validate(var_4)
    assert var_10 == 1
    var_11 = module_0.Field()
    var_12 = module_1.NeverMatch()
    var_13 = module_0.Field()
    var_14 = module_1.IfThenElse(var_11, var_12, var_13)
    var_15 = 1
    var_16 = var_14.validate(var_15)
    var_17 = module_1.NeverMatch()
    var_18 = module_0.Field()
    var_19 = module_1.NeverMatch()
    var_20 = module_1.IfThenElse(var_17, var_18, var_19)
    var_21 = 1
    var_22 = var_20.validate(var_21)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = [var_4, var_5]



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    assert var_4 == 5



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = [var_4, var_5]



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = [var_4, var_5]



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = [var_4, var_5]



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = var_3.one_of
    var_5 = var_3.one_of
    var_6 = len(var_5)
    assert var_6 == 2



# Parsed testcases at query #12
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #13
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'Never Match'
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    assert var_4 == 5



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()



# Parsed testcases at query #17
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)
    var_3 = module_0.Any()
    var_4 = [var_3]



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #21
#--------------------------


import typesystem.composites as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.OneOf(var_0)
    var_2 = module_1.Any()
    var_3 = module_1.Any()
    var_4 = [var_2, var_3]
    var_5 = module_0.OneOf(var_4)
    var_6 = module_1.Any()
    var_7 = module_1.Any()
    var_8 = module_1.Any()
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.OneOf(var_9)
    var_11 = module_1.Any()
    var_12 = module_1.Any()
    var_13 = module_1.Any()
    var_14 = module_1.Any()
    var_15 = [var_11, var_12, var_13, var_14]
    var_16 = module_0.OneOf(var_15)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()



# Parsed testcases at query #24
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = var_1.negated



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = '\n    Test the constructor of the Not class.\n    '
    var_1 = module_0.Any()
    var_2 = module_1.Not(var_1)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #32
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = module_0.Any()



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()



# Parsed testcases at query #35
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = var_1.negated
    var_3 = module_0.Field()
    var_4 = module_1.Not(var_3)
    var_5 = var_4.negated
    var_6 = module_0.Field()
    var_7 = 'test'
    var_8 = module_1.Not(var_6)
    var_9 = var_8.negated
    var_10 = module_0.Field()
    var_11 = module_1.Not(var_10)
    var_12 = var_11.negated
    var_13 = module_0.Field()
    var_14 = module_1.Not(var_13)
    var_15 = var_14.negated
    var_16 = module_0.Field()
    var_17 = module_1.Not(var_16)
    var_18 = var_17.negated
    var_19 = module_0.Field()
    var_20 = module_1.Not(var_19)
    var_21 = var_20.negated



# Parsed testcases at query #2
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #6
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_1.AllOf(var_3)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = [var_4, var_5]



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_0.Any()
    var_5 = module_0.Any()
    var_6 = module_0.Any()



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = var_3.all_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_3.all_of[var_6]
    var_8 = 1
    var_9 = var_3.all_of[var_8]



# Parsed testcases at query #12
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #13
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_1.OneOf(var_5)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #17
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #20
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()



# Parsed testcases at query #22
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_1.OneOf(var_3)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #28
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = var_1.negated



# Parsed testcases at query #30
#--------------------------


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'anything'
    var_2 = var_0.validate(var_1)



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



# Parsed testcases at query #32
#--------------------------


import typesystem.composites as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)
    var_2 = module_1.Any()
    var_3 = [var_2]
    var_4 = module_0.AllOf(var_3)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4.all_of[var_7]



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 5
    var_5 = var_3.validate(var_4)
    assert var_5 == 5



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = '\n    Test constructor of class OneOf.\n    '
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = [var_1, var_2]
    var_4 = module_1.OneOf(var_3)
    var_5 = module_0.Any()
    var_6 = module_0.Any()
    var_7 = [var_5, var_6]



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



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
    var_1 = module_0.Not(var_0)
    var_2 = var_1.negated



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.Not(var_0)
    var_2 = module_0.Any()



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



