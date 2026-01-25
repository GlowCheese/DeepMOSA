####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_ifthenelse_constructor_without_then_clause. Retrieved 4/5 statements.
# Partially parsed test_ifthenelse_constructor_without_else_clause. Retrieved 4/5 statements.
# Partially parsed test_ifthenelse_constructor_without_then_and_else_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
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
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, else_clause=var_1, **var_2)
    var_4 = var_3.if_clause
    var_5 = bool(var_3.if_clause == var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = var_3.else_clause
    var_8 = bool(var_3.else_clause == var_1)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, var_1, **var_2)
    var_4 = var_3.if_clause
    var_5 = bool(var_3.if_clause == var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = bool(var_3.then_clause == var_1)
    assert var_7 is True
    var_8 = var_3.else_clause

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
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_then_clause_when_if_clause_passes. Retrieved 1/5 statements.
# Partially parsed test_validate_else_clause_when_if_clause_fails. Retrieved 1/5 statements.
# Partially parsed test_validate_default_then_clause. Retrieved 1/4 statements.
# Partially parsed test_validate_default_else_clause. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '123'

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_matches_single_type. Retrieved 1/6 statements.
# Partially parsed test_validate_matches_single_type_with_different_value. Retrieved 1/6 statements.
# Partially parsed test_validate_raises_error_for_no_match. Retrieved 1/7 statements.
# Partially parsed test_validate_raises_error_for_multiple_matches. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 3.14

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Test2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = var_6.all_of
    var_8 = bool(var_6.all_of == [var_1, var_3])
    assert var_8 is True
    var_9 = var_6.title
    assert var_9 == ''
    var_10 = var_6.description
    assert var_10 == ''
    var_11 = var_6.allow_null
    assert var_11 is False
    var_12 = var_6.read_only
    assert var_12 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Test1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'not a list'
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_1.OneOf(var_2, **var_3)
    var_5 = var_4.one_of
    var_6 = bool(var_4.one_of == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.title
    assert var_7 == ''
    var_8 = var_4.description
    assert var_8 == ''
    var_9 = var_4.allow_null
    assert var_9 is False
    var_10 = var_4.read_only
    assert var_10 is False



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Number(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'
    var_9 = 123
    var_10 = var_6.validate(var_9)
    assert var_10 == 123

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Number(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = []
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_0.Any()
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.OneOf(var_3, **var_4)
    var_6 = 'test'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.OneOf(var_0, **var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_not_validate_with_matching_negated_field. Retrieved 6/8 statements.
# Partially parsed test_not_validate_with_non_matching_negated_field. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = (var_1, var_1)
    var_3 = {}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = 'test_value'
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = module_1.Not(var_0, **var_4)
    var_6 = 'test_value'
    var_7 = var_5.validate(var_6)
    assert var_7 == 'test_value'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_never_match_constructor_without_allow_null. Retrieved 1/2 statements.
# Partially parsed test_never_match_constructor_with_default. Retrieved 2/4 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False
    var_3 = var_1.title
    assert var_3 == ''
    var_4 = var_1.description
    assert var_4 == ''
    var_5 = var_1.read_only
    assert var_5 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'title'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.title
    assert var_4 == 'Test Title'

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Description'
    var_1 = 'description'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.description
    assert var_4 == 'Test Description'

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_multiple_matches. Retrieved 1/10 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 3.14
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'test'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------




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
    var_7 = var_2.allow_null
    assert var_7 is False
    var_8 = var_2.read_only
    assert var_8 is False

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



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 1.5
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = '123'
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = var_4.all_of
    var_6 = bool(var_4.all_of == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.title
    assert var_7 == ''
    var_8 = var_4.description
    assert var_8 == ''
    var_9 = bool(not var_4.allow_null)
    assert var_9 is True
    var_10 = bool(not var_4.read_only)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.AllOf(var_0, **var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ifthenelse_constructor_with_default_then_else_clauses. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
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
    var_11 = var_4.allow_null
    assert var_11 is False
    var_12 = var_4.read_only
    assert var_12 is False

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
    var_7 = var_2.allow_null
    assert var_7 is False
    var_8 = var_2.read_only
    assert var_8 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_1.OneOf(var_2, **var_3)
    var_5 = var_4.one_of
    var_6 = bool(var_4.one_of == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.title
    assert var_7 == ''
    var_8 = var_4.description
    assert var_8 == ''
    var_9 = var_4.allow_null
    assert var_9 is False
    var_10 = var_4.read_only
    assert var_10 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_9 = var_8.title
    assert var_9 == 'Test Title'
    var_10 = var_8.description
    assert var_10 == 'Test Description'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_never_match_constructor_without_allow_null. Retrieved 1/2 statements.
# Partially parsed test_never_match_constructor_with_title_and_description. Retrieved 3/4 statements.
# Partially parsed test_never_match_constructor_with_read_only. Retrieved 2/3 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False
    var_3 = var_1.title
    assert var_3 == ''
    var_4 = var_1.description
    assert var_4 == ''
    var_5 = var_1.read_only
    assert var_5 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.NeverMatch(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Title'
    var_7 = var_5.description
    assert var_7 == 'Test Description'
    var_8 = var_5.allow_null
    assert var_8 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True
    var_5 = var_3.allow_null
    assert var_5 is False



