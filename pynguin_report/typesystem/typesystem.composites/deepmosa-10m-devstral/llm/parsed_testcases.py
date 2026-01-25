####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = 'valid_value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'valid_value'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = 'invalid_value'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_negated_field_validation_with_error. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = None
    var_4 = 'error'
    var_5 = (var_3, var_4)
    var_6 = 'test_value'
    var_7 = var_2.validate(var_6)
    assert var_7 == 'test_value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_returns_value_when_error_exists. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_value'



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
    var_9 = var_4.allow_null
    assert var_9 is False
    var_10 = var_4.read_only
    assert var_10 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = bool(False)
    assert var_7 is True

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
    var_8 = module_1.AllOf(var_2, **var_7)
    var_9 = var_8.all_of
    var_10 = bool(var_8.all_of == [var_0, var_1])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Test Title'
    var_12 = var_8.description
    assert var_12 == 'Test Description'
    var_13 = var_8.allow_null
    assert var_13 is False
    var_14 = var_8.read_only
    assert var_14 is False



# Parsed testcases at query #5
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
    var_11 = var_4.allow_null
    assert var_11 is False

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
    var_9 = var_3.allow_null
    assert var_9 is False

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
    var_9 = var_3.allow_null
    assert var_9 is False

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



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Test Description'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.OneOf(var_2, **var_7)
    var_9 = var_8.one_of
    var_10 = bool(var_8.one_of == [var_0, var_1])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Test'
    var_12 = var_8.description
    assert var_12 == 'Test Description'
    var_13 = var_8.allow_null
    assert var_13 is False
    var_14 = var_8.read_only
    assert var_14 is False



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




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test'
    var_2 = 'Test Description'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_1.Not(var_0, **var_5)
    var_7 = var_6.negated
    var_8 = bool(var_6.negated == var_0)
    assert var_8 is True
    var_9 = var_6.title
    assert var_9 == 'Test'
    var_10 = var_6.description
    assert var_10 == 'Test Description'
    var_11 = var_6.allow_null
    assert var_11 is False
    var_12 = var_6.read_only
    assert var_12 is False



# Parsed testcases at query #9
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
    var_7 = 3.14
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



# Parsed testcases at query #2
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.NeverMatch(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test'
    var_7 = var_5.description
    assert var_7 == 'Test Description'
    var_8 = var_5.allow_null
    assert var_8 is False
    var_9 = var_5.read_only
    assert var_9 is False



# Parsed testcases at query #3
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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = 'Test Description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.AllOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test'
    var_9 = var_7.description
    assert var_9 == 'Test Description'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 'Test'
    var_4 = 'Test Description'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.OneOf(var_2, **var_7)
    var_9 = var_8.one_of
    var_10 = bool(var_8.one_of == [var_0, var_1])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Test'
    var_12 = var_8.description
    assert var_12 == 'Test Description'
    var_13 = var_8.allow_null
    assert var_13 is False
    var_14 = var_8.read_only
    assert var_14 is False



# Parsed testcases at query #6
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
    var_1 = [var_0]
    var_2 = 'Test Title'
    var_3 = 'Test Description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.OneOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Title'
    var_9 = var_7.description
    assert var_9 == 'Test Description'



# Parsed testcases at query #7
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)



