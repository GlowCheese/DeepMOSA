####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_success_when_negated_fails. Retrieved 1/9 statements.
# Partially parsed test_validate_raises_error_when_negated_succeeds. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'some_value'

def test_case_0():
    var_0 = 'some_value'
    var_1 = 'ValueError not raised'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test Title'
    var_4 = 'Test Description'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.OneOf(var_2, **var_7)
    var_9 = var_8.one_of
    var_10 = bool(var_8.one_of == [var_0, var_1])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Test Title'
    var_12 = var_8.description
    assert var_12 == 'Test Description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.OneOf(var_1, **var_2)
    var_4 = var_3.one_of
    var_5 = bool(var_3.one_of == [var_0])
    assert var_5 is True
    var_6 = var_3.title
    assert var_6 == ''
    var_7 = var_3.description
    assert var_7 == ''

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = 'OneOf should raise AssertionError if allow_null is passed in kwargs'
    var_7 = AssertionError(var_6)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Not Field'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.Not(var_1, **var_4)
    var_6 = var_5.negated
    var_7 = bool(var_5.negated == var_1)
    assert var_7 is True
    var_8 = var_5.title
    assert var_8 == 'Not Field'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Desc'
    var_2 = True
    var_3 = 'description'
    var_4 = 'read_only'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_1.Not(var_0, **var_5)
    var_7 = var_6.description
    assert var_7 == 'Desc'
    var_8 = var_6.read_only
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field B'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'Combined Field'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_1.AllOf(var_4, **var_7)
    var_9 = var_8.all_of
    var_10 = bool(var_8.all_of == [var_1, var_3])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Combined Field'
    var_12 = var_8.allow_null
    assert var_12 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = var_4.description
    assert var_5 == ''
    var_6 = var_4.read_only
    assert var_6 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_if_then_else_init_defaults. Retrieved 4/6 statements.
# Partially parsed test_if_then_else_init_no_else_clause. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = 'test'
    var_4 = 'title'
    var_5 = {var_4: var_3}
    var_6 = module_1.IfThenElse(var_0, var_1, var_2, **var_5)
    var_7 = var_6.if_clause
    var_8 = bool(var_6.if_clause == var_0)
    assert var_8 is True
    var_9 = var_6.then_clause
    var_10 = bool(var_6.then_clause == var_1)
    assert var_10 is True
    var_11 = var_6.else_clause
    var_12 = bool(var_6.else_clause == var_2)
    assert var_12 is True
    var_13 = var_6.title
    assert var_13 == 'test'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause == var_0)
    assert var_4 is True
    var_5 = var_2.then_clause
    var_6 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, var_1, **var_2)
    var_4 = var_3.then_clause
    var_5 = bool(var_3.then_clause == var_1)
    assert var_5 is True
    var_6 = var_3.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = 'Should have raised AssertionError for allow_null in kwargs'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #6
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A test description'
    var_2 = 10
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'default'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.NeverMatch(**var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Field'
    var_9 = var_7.description
    assert var_9 == 'A test description'
    var_10 = var_7.default
    assert var_10 == 10
    var_11 = var_7.allow_null
    assert var_11 is False
    var_12 = var_7.read_only
    assert var_12 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'title'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = None
    var_5 = 'description'
    var_6 = {var_5: var_4}
    var_7 = module_0.NeverMatch(**var_6)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_no_match_raises_error. Retrieved 1/11 statements.
# Partially parsed test_validate_multiple_matches_raises_error. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'non_existent'

def test_case_0():
    var_0 = 'anything'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_no_match_raises_error. Retrieved 1/11 statements.
# Partially parsed test_validate_multiple_matches_raises_error. Retrieved 1/12 statements.
# Partially parsed test_validate_with_mixed_results. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'unmatched'

def test_case_0():
    var_0 = 'any'

def test_case_0():
    var_0 = 'match'
    var_1 = 'fail'



# Parsed testcases at query #3
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'Test Description'
    var_2 = 123
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'default'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.NeverMatch(**var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Field'
    var_9 = var_7.description
    assert var_9 == 'Test Description'
    var_10 = var_7.default
    assert var_10 == 123
    var_11 = var_7.allow_null
    assert var_11 is False
    var_12 = var_7.read_only
    assert var_12 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Base'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Not Field'
    var_3 = 'Desc'
    var_4 = True
    var_5 = 'title'
    var_6 = 'description'
    var_7 = 'read_only'
    var_8 = {var_5: var_2, var_6: var_3, var_7: var_4}
    var_9 = module_1.Not(var_1, **var_8)
    var_10 = var_9.negated
    var_11 = bool(var_9.negated == var_1)
    assert var_11 is True
    var_12 = var_9.title
    assert var_12 == 'Not Field'
    var_13 = var_9.description
    assert var_13 == 'Desc'
    var_14 = var_9.read_only
    assert var_14 is True
    var_15 = var_9.errors
    var_16 = bool(var_9.errors == {'negated': 'Must not match.'})
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = 'Not constructor should have raised AssertionError when allow_null is provided in kwargs'
    var_6 = AssertionError(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = var_2.negated
    var_4 = bool(var_2.negated == var_0)
    assert var_4 is True
    var_5 = var_2.title
    assert var_5 == ''
    var_6 = var_2.description
    assert var_6 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_success_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_error_no_match. Retrieved 1/11 statements.
# Partially parsed test_validate_error_multiple_matches. Retrieved 1/12 statements.
# Partially parsed test_validate_returns_correct_candidate_value. Retrieved 3/35 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'unmatched'

def test_case_0():
    var_0 = 'any'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'target'



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field B'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'OneOf Field'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_1.OneOf(var_4, **var_7)
    var_9 = var_8.one_of
    var_10 = bool(var_8.one_of == [var_1, var_3])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'OneOf Field'
    var_12 = var_8.description
    assert var_12 == ''
    var_13 = var_8.allow_null
    assert var_13 is False
    var_14 = var_8.read_only
    assert var_14 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.OneOf(var_2, **var_5)
    var_7 = 'Should have raised AssertionError due to allow_null in kwargs'
    var_8 = AssertionError(var_7)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'B'
    var_3 = 'Desc'
    var_4 = True
    var_5 = module_0.Field(title=var_2, description=var_3, read_only=var_4)
    var_6 = [var_1, var_5]
    var_7 = 'Main Desc'
    var_8 = 'description'
    var_9 = {var_8: var_7}
    var_10 = module_1.OneOf(var_6, **var_9)
    var_11 = var_10.description
    assert var_11 == 'Main Desc'
    var_12 = var_10.one_of[1].description
    assert var_12 == 'Desc'
    var_13 = var_10.one_of[1].read_only
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_if_then_else_constructor_defaults. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = {}
    var_4 = module_1.IfThenElse(var_0, var_1, var_2, **var_3)
    var_5 = var_4.if_clause
    var_6 = bool(var_4.if_clause == var_0)
    assert var_6 is True
    var_7 = var_4.then_clause
    var_8 = bool(var_4.then_clause == var_1)
    assert var_8 is True
    var_9 = var_4.else_clause
    var_10 = bool(var_4.else_clause == var_2)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause == var_0)
    assert var_4 is True
    var_5 = var_2.then_clause
    var_6 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'Test Title'
    var_3 = 'Test Desc'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.IfThenElse(var_0, var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Title'
    var_9 = var_7.description
    assert var_9 == 'Test Desc'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = var_4.read_only
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field B'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 'AllOf Field'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_1.AllOf(var_4, **var_7)
    var_9 = var_8.all_of
    var_10 = bool(var_8.all_of == [var_1, var_3])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'AllOf Field'
    var_12 = var_8.allow_null
    assert var_12 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = 'Should have raised AssertionError due to allow_null in kwargs'
    var_8 = AssertionError(var_7)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field A'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'Description'
    var_4 = 'description'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = var_6.description
    assert var_7 == 'Description'
    var_8 = var_6.read_only
    assert var_8 is False



