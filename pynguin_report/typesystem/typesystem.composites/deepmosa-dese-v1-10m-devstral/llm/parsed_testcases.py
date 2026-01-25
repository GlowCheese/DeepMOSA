####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_validation_fails. Retrieved 7/8 statements.
# Partially parsed test_validate_raises_error_when_negated_validation_succeeds. Retrieved 7/9 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = module_1.Not(var_0)
    var_5 = 'test_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test_value'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid'
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = module_1.Not(var_0)
    var_5 = 'test_value'
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_all_of_constructor. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_never_match_constructor. Retrieved 3/4 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test Description'
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #5
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
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_if_then_else_constructor_with_default_then_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_with_default_else_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_with_default_then_and_else_clauses. Retrieved 4/6 statements.


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
    var_3 = var_2.then_clause

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
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = 'Test'
    var_2 = 'Test Description'
    var_3 = module_1.IfThenElse(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Test Description'
    var_5 = module_1.OneOf(var_2)



# Parsed testcases at query #11
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_if_then_else_constructor_with_missing_then_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_with_missing_else_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_with_missing_both_clauses. Retrieved 4/6 statements.


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
    var_3 = var_2.then_clause

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
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_if_then_else_constructor_without_then_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_without_else_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_without_then_and_else_clauses. Retrieved 4/6 statements.


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
    var_3 = var_2.then_clause

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
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #19
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
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Custom Title'
    var_3 = 'Custom Description'
    var_4 = module_1.OneOf(var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #21
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
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = 'Description'
    var_4 = module_1.OneOf(var_1)



# Parsed testcases at query #22
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
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test'
    var_2 = 'Desc'
    var_3 = True
    var_4 = module_1.Not(var_0)
    var_5 = 'default'
    var_6 = hasattr(var_4, var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #29
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_never_match_constructor_initialization. Retrieved 3/4 statements.
# Partially parsed test_never_match_constructor_with_default. Retrieved 2/3 statements.
# Partially parsed test_never_match_constructor_with_callable_default. Retrieved 3/4 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test Description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'callable_default'
    var_1 = lambda : var_0
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #32
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #33
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #34
#--------------------------




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
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_never_match_constructor. Retrieved 3/4 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test description'
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #38
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_if_then_else_constructor_without_then_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_without_else_clause. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_without_then_and_else_clauses. Retrieved 4/6 statements.


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
    var_3 = var_2.then_clause

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
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #40
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #41
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = 3.14
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = '123'
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ifthenelse_constructor_with_missing_then_clause. Retrieved 4/5 statements.
# Partially parsed test_ifthenelse_constructor_with_missing_else_clause. Retrieved 4/5 statements.
# Partially parsed test_ifthenelse_constructor_with_missing_both_clauses. Retrieved 4/6 statements.


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
    var_3 = var_2.then_clause

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
    var_1 = module_1.IfThenElse(var_0)
    var_2 = var_1.then_clause
    var_3 = var_1.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_never_match_constructor_without_allow_null. Retrieved 1/2 statements.
# Partially parsed test_never_match_constructor_with_default. Retrieved 2/4 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Description'
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #7
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = 'Test'
    var_2 = 'Test description'
    var_3 = module_0.OneOf(var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #11
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #14
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #15
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #18
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #19
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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test'
    var_2 = 'Test Description'
    var_3 = True
    var_4 = module_1.Not(var_0)



# Parsed testcases at query #20
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_ifthenelse_constructor_with_missing_then_clause. Retrieved 4/6 statements.
# Partially parsed test_ifthenelse_constructor_with_missing_else_clause. Retrieved 4/5 statements.


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
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #22
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #23
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #24
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_if_then_else_init_with_allow_null_raises_assertion_error. Retrieved 2/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #27
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #29
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #32
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test description'
    var_2 = module_0.NeverMatch()



# Parsed testcases at query #33
#--------------------------




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
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #37
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
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Custom Title'
    var_3 = 'Custom Description'
    var_4 = module_1.AllOf(var_1)



# Parsed testcases at query #38
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #39
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.OneOf(var_0)



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #41
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



