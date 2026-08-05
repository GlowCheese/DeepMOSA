####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_not_validate_success. Retrieved 1/16 statements.
# Partially parsed test_not_validate_failure. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'allowed_value'

def test_case_0():
    var_0 = 'forbidden_value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_if_then_else_constructor_with_defaults. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'If'
    var_1 = module_0.Field(title=var_0)
    var_2 = module_0.Any()
    var_3 = module_0.Any()
    var_4 = 'Conditional'
    var_5 = module_1.IfThenElse(var_1, var_2, var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'If'
    var_1 = module_0.Field(title=var_0)
    var_2 = module_1.IfThenElse(var_1)
    var_3 = var_2.then_clause
    var_4 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = "Should have raised AssertionError due to 'allow_null' in kwargs"
    var_4 = AssertionError(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test Desc'
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #3
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A test description'
    var_2 = 123
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #4
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'desc'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = 'Desc 2'
    var_4 = module_0.Field(title=var_2, description=var_3)
    var_5 = [var_1, var_4]
    var_6 = 'AllOf Title'
    var_7 = module_1.AllOf(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)
    var_4 = 'Should have raised an error because allow_null is forbidden in AllOf kwargs'
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test Description'
    var_3 = True
    var_4 = module_1.AllOf(var_1)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = 'Desc A'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'Field B'
    var_4 = 'Desc B'
    var_5 = module_0.Field(title=var_3, description=var_4)
    var_6 = [var_2, var_5]
    var_7 = 'Union Field'
    var_8 = 'A union of fields'
    var_9 = module_1.OneOf(var_6)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_not_constructor_sets_attributes. Retrieved 3/8 statements.
# Partially parsed test_not_constructor_raises_error_on_allow_null_in_kwargs. Retrieved 2/14 statements.
# Failed to parse test_not_constructor_initializes_with_default_params.


def test_case_0():
    var_0 = 'Negated'
    var_1 = 'Not Field'
    var_2 = 'Test Desc'

def test_case_0():
    var_0 = True
    var_1 = 'Did not raise AssertionError'



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = "AssertionError was not raised when 'allow_null' is in kwargs"
    var_4 = AssertionError(var_3)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_if_then_else_constructor_defaults. Retrieved 4/6 statements.
# Partially parsed test_if_then_else_constructor_valid_assignment. Retrieved 4/5 statements.


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
    var_1 = 'Test Title'
    var_2 = 'Test Desc'
    var_3 = module_1.IfThenElse(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Any()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = var_2.else_clause



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_oneof_init_assertion_fails_when_allow_null_is_provided. Retrieved 1/8 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_if_then_else_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 4/7 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = lambda : var_2



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_not_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/12 statements.
# Partially parsed test_not_init_success_without_allow_null. Retrieved 1/8 statements.


def test_case_0():
    var_0 = True
    var_1 = "Not.__init__ should have raised AssertionError when 'allow_null' is in kwargs"
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 'Test Title'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_all_of_init_raises_assertion_error_when_allow_null_is_passed. Retrieved 3/13 statements.


def test_case_0():
    var_0 = True
    var_1 = "AssertionError was not raised when 'allow_null' was passed to AllOf.__init__"
    var_2 = AssertionError(var_1)



# Parsed testcases at query #16
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A test description'
    var_2 = 123
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'some value'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_allof_constructor_handles_default_value. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'f2'
    var_3 = 'desc'
    var_4 = module_0.Field(title=var_2, description=var_3)
    var_5 = [var_1, var_4]
    var_6 = 'combined'
    var_7 = 'all of'
    var_8 = module_1.AllOf(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)
    var_4 = 'Should have raised AssertionError because allow_null is forbidden in AllOf kwargs'
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'some_default'
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Not Field'
    var_3 = 'A test field'
    var_4 = module_1.Not(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated'
    var_1 = module_0.Field(title=var_0)
    var_2 = True
    var_3 = module_1.Not(var_1)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated'
    var_1 = module_0.Field(title=var_0)
    var_2 = True
    var_3 = module_1.Not(var_1)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'f2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'composite'
    var_6 = 'desc'
    var_7 = module_1.AllOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'Union Field'
    var_6 = 'Test Description'
    var_7 = module_1.OneOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = module_1.OneOf(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_oneof_init_raises_assertion_error_when_allow_null_is_passed. Retrieved 9/12 statements.
# Partially parsed test_oneof_init_assertion_error. Retrieved 5/8 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)
    var_4 = lambda : var_3
    var_5 = module_0.Field()
    var_6 = [var_5]
    var_7 = True
    var_8 = module_1.OneOf(var_6)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)
    var_4 = 'Assertion should have been raised'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_not_init_asserts_no_allow_null_in_kwargs. Retrieved 3/12 statements.
# Partially parsed test_not_init_success_without_allow_null_in_kwargs. Retrieved 1/8 statements.


def test_case_0():
    var_0 = True
    var_1 = "The assertion 'allow_null' not in kwargs should have failed."
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 'Test Field'



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'Union Field'
    var_6 = module_1.OneOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)
    var_5 = 'OneOf should raise AssertionError when allow_null is passed in kwargs'
    var_6 = AssertionError(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'Description'
    var_4 = module_1.OneOf(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_if_then_else_constructor_with_defaults. Retrieved 4/6 statements.
# Partially parsed test_if_then_else_constructor_with_only_then. Retrieved 4/5 statements.
# Partially parsed test_if_then_else_constructor_with_only_else. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = 'Test'
    var_4 = 'Desc'
    var_5 = module_1.IfThenElse(var_0, var_1, var_2)

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
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = "Should have raised AssertionError due to 'allow_null' in kwargs"
    var_4 = AssertionError(var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_allof_init_raises_error_on_allow_null_in_kwargs. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #26
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Child 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Child 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'AllOf Field'
    var_6 = 'Test Description'
    var_7 = module_1.AllOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)
    var_4 = 'Should have raised AssertionError because allow_null is not allowed in kwargs'
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_if_then_else_init_raises_assertion_error_on_allow_null_in_kwargs. Retrieved 4/8 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = lambda : var_2



# Parsed testcases at query #29
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Never Field'
    var_1 = 'A field that never matches'
    var_2 = True
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_oneof_init_raises_assertion_error_on_allow_null_in_kwargs. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_not_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/12 statements.


def test_case_0():
    var_0 = True
    var_1 = "Expected AssertionError when 'allow_null' is passed to Not.__init__"
    var_2 = AssertionError(var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_oneof_init_raises_assertion_error_on_allow_null. Retrieved 3/9 statements.
# Failed to parse test_oneof_init_success_without_allow_null.


def test_case_0():
    var_0 = True
    var_1 = 'AssertionError was not raised despite passing allow_null in kwargs'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)



# Parsed testcases at query #34
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_if_then_else_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 6/11 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = lambda : var_2
    var_4 = True
    var_5 = module_1.IfThenElse(var_0)



# Parsed testcases at query #36
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)
    var_3 = "AssertionError was not raised when 'allow_null' was passed to AllOf.__init__"
    var_4 = AssertionError(var_3)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = 'Test Field'
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_allof_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/13 statements.


def test_case_0():
    var_0 = True
    var_1 = "Expected assertion error for 'allow_null' in kwargs, but none was raised."
    var_2 = AssertionError(var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_if_then_else_init_raises_assertion_error_on_allow_null_in_kwargs. Retrieved 6/12 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)
    var_3 = lambda : var_2
    var_4 = True
    var_5 = module_1.IfThenElse(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #39
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)
    var_3 = "Failed to raise AssertionError when 'allow_null' is passed in kwargs"
    var_4 = AssertionError(var_3)



# Parsed testcases at query #40
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A description'
    var_2 = True
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_when_allow_null_is_passed. Retrieved 3/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 'AssertionError was not raised'
    var_2 = AssertionError(var_1)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_no_match. Retrieved 1/12 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'valid'

def test_case_0():
    var_0 = 'invalid'

def test_case_0():
    var_0 = 'any'



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'Union Field'
    var_6 = 'A test field'
    var_7 = module_1.OneOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)
    var_4 = "Should have raised AssertionError due to 'allow_null' in kwargs"
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'AllOf'
    var_4 = 'Test Description'
    var_5 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)
    var_4 = 'AllOf should raise AssertionError when allow_null is provided in kwargs'
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_all_of_init_raises_assertion_error_on_allow_null_in_kwargs. Retrieved 3/13 statements.


def test_case_0():
    var_0 = True
    var_1 = 'AssertionError was not raised'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.AllOf(var_1)



# Parsed testcases at query #6
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'Test Description'
    var_2 = 123
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Inner'
    var_1 = 'Desc'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'NotField'
    var_4 = 'NotDesc'
    var_5 = module_1.Not(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.Not(var_0)
    var_3 = "Should have raised AssertionError due to 'allow_null' in kwargs"
    var_4 = AssertionError(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_1.Not(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_if_then_else_constructor_defaults. Retrieved 4/6 statements.


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
    var_1 = module_0.Any()
    var_2 = module_0.Field()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Conditional'
    var_2 = 'Test'
    var_3 = module_1.IfThenElse(var_0)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_1.IfThenElse(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_not_init_assertion_fails_when_allow_null_is_provided. Retrieved 6/11 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_null'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = 'Assertion failed to trigger'



# Parsed testcases at query #10
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #11
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.AllOf(var_0)
    var_3 = 'The constructor should have raised an AssertionError when allow_null is provided.'
    var_4 = AssertionError(var_3)

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = 'Test Field'
    var_2 = module_0.AllOf(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A description'
    var_2 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_no_match. Retrieved 1/11 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'unmatched'

def test_case_0():
    var_0 = 'any'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_one_of_multiple_matches. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'val1'
    var_1 = 'val2'
    var_2 = 'some_value'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 'Expected assertion error was not raised'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #16
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = module_0.NeverMatch()

def test_case_0():
    var_0 = 'allow_null'
    var_1 = True
    var_2 = {var_0: var_1}



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = 'allow_null'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = 'AssertionError not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_1.OneOf(var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/12 statements.
# Failed to parse test_one_of_init_succeeds_when_allow_null_is_not_provided.


def test_case_0():
    var_0 = True
    var_1 = "AssertionError was not raised despite 'allow_null' being in kwargs"
    var_2 = AssertionError(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_oneof_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_never_match_validate_always_fails. Retrieved 4/7 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = 'any value'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = 'Description 2'
    var_4 = module_0.Field(title=var_2, description=var_3)
    var_5 = [var_1, var_4]
    var_6 = 'Composite Field'
    var_7 = module_1.AllOf(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = 'default_val'
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = [var_2]
    var_4 = module_1.AllOf(var_3)



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'AllOf Field'
    var_6 = 'Description'
    var_7 = module_1.AllOf(var_4)

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
    var_2 = module_1.AllOf(var_1)



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test Title'
    var_4 = 'Test Description'
    var_5 = module_1.OneOf(var_2)

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
    var_2 = module_1.OneOf(var_1)



# Parsed testcases at query #25
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A test description'
    var_2 = 10
    var_3 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = module_0.NeverMatch()



# Parsed testcases at query #26
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = module_0.NeverMatch()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 3/13 statements.
# Partially parsed test_one_of_init_success_without_allow_null. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True
    var_1 = "Expected assertion error because 'allow_null' was passed in kwargs"
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 'Test Field'



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'AllOf Field'
    var_6 = 'Test Description'
    var_7 = module_1.AllOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'some_default'
    var_4 = True
    var_5 = module_1.AllOf(var_2)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'f2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'test_allof'
    var_6 = 'test_desc'
    var_7 = module_1.AllOf(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.AllOf(var_2)
    var_5 = 'Should have raised AssertionError due to allow_null in kwargs'
    var_6 = AssertionError(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = module_1.AllOf(var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_on_allow_null. Retrieved 1/10 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #31
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.NeverMatch()



