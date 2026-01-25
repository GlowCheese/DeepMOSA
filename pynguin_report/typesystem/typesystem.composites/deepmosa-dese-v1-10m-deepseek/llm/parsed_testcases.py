####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_Not_validate_passes_when_negated_fails. Retrieved 6/10 statements.
# Partially parsed test_Not_validate_fails_when_negated_passes. Retrieved 5/7 statements.
# Partially parsed test_Not_validate_passes_with_null_value. Retrieved 5/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'error'
    var_3 = module_1.Not(var_0)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_1.Not(var_0)
    var_3 = 'test'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = True
    var_3 = module_1.Not(var_0)
    var_4 = var_3.validate(var_1)
    assert var_4 is None



# Parsed testcases at query #2
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = True
    var_3 = module_0.NeverMatch()



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.OneOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_1.IfThenElse(var_0, else_clause=var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_1.IfThenElse(var_0, var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.IfThenElse(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = 'allow_null should not be allowed in constructor'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'title'
    var_2 = 'description'
    var_3 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #11
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = module_0.NeverMatch()
    var_3 = 'default'
    var_4 = hasattr(var_2, var_3)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test Title'
    var_4 = 'Test Description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_1.AllOf(var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ifthenelse_constructor_with_only_if_clause. Retrieved 4/6 statements.
# Partially parsed test_ifthenelse_constructor_with_then_clause. Retrieved 4/5 statements.
# Partially parsed test_ifthenelse_constructor_with_else_clause. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_1.IfThenElse(var_0, else_clause=var_1)
    var_3 = var_2.then_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'test description'
    var_4 = module_1.AllOf(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Test Description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = False
    var_3 = module_1.IfThenElse(var_1)



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #26
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'title'
    var_2 = 'description'
    var_3 = True
    var_4 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'title'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = [var_0]
    var_5 = module_1.OneOf(var_4, **var_3)



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #31
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #32
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Example'
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_ifthenelse_constructor_with_default_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'default'
    var_5 = hasattr(var_3, var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_null'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = [var_0]
    var_5 = module_1.OneOf(var_4, **var_3)



# Parsed testcases at query #37
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #38
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #39
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = 'allow_null'
    var_5 = hasattr(var_3, var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Test description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_exactly_one_match. Retrieved 11/13 statements.
# Partially parsed test_validate_no_match. Retrieved 10/13 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 7/10 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = None
    var_3 = 'error'
    var_4 = (var_2, var_3)
    var_5 = module_0.Field()
    var_6 = 2
    var_7 = (var_2, var_3)
    var_8 = [var_0, var_5]
    var_9 = module_1.OneOf(var_8)
    var_10 = var_9.validate(var_1)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = module_0.Field()
    var_5 = (var_1, var_2)
    var_6 = [var_0, var_4]
    var_7 = module_1.OneOf(var_6)
    var_8 = 1
    var_9 = var_7.validate(var_8)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_0.Field()
    var_3 = [var_0, var_2]
    var_4 = module_1.OneOf(var_3)
    var_5 = 1
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'title'
    var_2 = 'description'
    var_3 = True
    var_4 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #4
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = module_0.NeverMatch()
    var_3 = 'default'
    var_4 = hasattr(var_2, var_3)

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_if_then_else_constructor_default_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = "IfThenElse should not allow 'allow_null' keyword argument"
    var_4 = AssertionError(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #6
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'test'
    var_4 = 'test description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.OneOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #15
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_if_then_else_init_without_allow_null.




# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = [var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_IfThenElse_constructor_default_then_else_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #26
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #31
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = True
    var_3 = module_0.NeverMatch()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_allof_constructor_inherits_from_field. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_if_then_else_constructor_with_default_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'title'
    var_2 = 'description'
    var_3 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #35
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = True
    var_3 = module_0.NeverMatch()



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #37
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)
    var_3 = 'Expected AssertionError'
    var_4 = AssertionError(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #38
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #39
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.IfThenElse(var_0)



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #41
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.AllOf(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)



