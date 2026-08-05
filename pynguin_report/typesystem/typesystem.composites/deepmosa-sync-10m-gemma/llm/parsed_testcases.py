####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_if_then_else_constructor_defaults. Retrieved 4/6 statements.
# Partially parsed test_if_then_else_constructor_only_then. Retrieved 4/5 statements.


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
    var_1 = 'Test Title'
    var_2 = 'Test Description'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_1.IfThenElse(var_0, **var_5)
    var_7 = var_6.title
    assert var_7 == 'Test Title'
    var_8 = var_6.description
    assert var_8 == 'Test Description'

import typesystem.fields as module_0
import typesystem.composites as module_1
import builtins as module_2

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = 'Should have failed'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_2.Exception(*var_6, **var_7)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = var_4.read_only
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_if_true. Retrieved 4/9 statements.
# Partially parsed test_validate_if_false. Retrieved 4/9 statements.
# Partially parsed test_validate_with_default_clauses. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'some_input'

def test_case_0():
    var_0 = False
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'some_input'

def test_case_0():
    var_0 = True



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated Field'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Not Field'
    var_3 = 'Description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.Not(var_1, **var_6)
    var_8 = var_7.negated
    var_9 = bool(var_7.negated == var_1)
    assert var_9 is True
    var_10 = var_7.title
    assert var_10 == 'Not Field'
    var_11 = var_7.description
    assert var_11 == 'Description'
    var_12 = var_7.errors['negated']
    assert var_12 == 'Must not match.'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = 'Not constructor should raise AssertionError when allow_null is provided in kwargs'
    var_6 = AssertionError(var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = var_4.read_only
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'AllOfField'
    var_4 = 'Test Description'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.AllOf(var_2, **var_7)
    var_9 = var_8.all_of
    var_10 = bool(var_8.all_of == [var_1])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'AllOfField'
    var_12 = var_8.description
    assert var_12 == 'Test Description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = var_4.allow_null
    assert var_5 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = 'AllOf should raise AssertionError when allow_null is passed in kwargs'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_success_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_raises_no_match. Retrieved 1/11 statements.
# Partially parsed test_validate_raises_multiple_matches. Retrieved 1/12 statements.
# Partially parsed test_validate_returns_transformed_value. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'unmatched'

def test_case_0():
    var_0 = 'any'

def test_case_0():
    var_0 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_one_of_exactly_one_match_returns_candidate. Retrieved 4/33 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = None
    var_2 = 'error'
    var_3 = 'input_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_one_of_single_match_returns_candidate. Retrieved 4/31 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = None
    var_2 = 'error'
    var_3 = 'some_value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_not_raises_error_when_subfield_matches. Retrieved 2/12 statements.
# Partially parsed test_validate_not_returns_value_when_subfield_does_not_match. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'some_value'
    var_1 = str(var_0)
    var_2 = 'negated'
    var_3 = bool('negated' in var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 'some_value'



# Parsed testcases at query #9
#--------------------------




import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'A description'
    var_2 = 123
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'default'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.NeverMatch(**var_6)
    var_8 = var_7.title
    var_9 = bool(var_7.title == var_0)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_1)
    assert var_11 is True
    var_12 = var_7.default
    var_13 = bool(var_7.default == var_2)
    assert var_13 is True
    var_14 = var_7.allow_null
    assert var_14 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_one_of_exactly_one_match. Retrieved 5/36 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = None
    var_2 = 'fail'
    var_3 = 'error'
    var_4 = 'some_value'



# Parsed testcases at query #11
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
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_1.OneOf(var_4, **var_7)
    var_9 = var_8.one_of
    var_10 = bool(var_8.one_of == [var_1, var_3])
    assert var_10 is True
    var_11 = var_8.title
    assert var_11 == 'Union Field'
    var_12 = var_8.allow_null
    assert var_12 is False

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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.OneOf(var_1, **var_2)
    var_4 = var_3.errors['no_match']
    assert var_4 == 'Did not match any valid type.'
    var_5 = var_3.errors['multiple_matches']
    assert var_5 == 'Matched more than one type.'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_not_with_custom_validation_error_method. Retrieved 1/11 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = None
    var_4 = 'Error'
    var_5 = (var_3, var_4)
    var_6 = lambda self, v: var_5
    var_7 = {var_2: var_6}
    var_8 = type(var_0, var_1, var_7)
    var_9 = var_8()
    var_10 = {}
    var_11 = module_0.Not(var_9, **var_10)
    var_12 = 'some_value'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'some_value'

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = None
    var_4 = lambda self, v: (v, var_3)
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = var_6()
    var_8 = {}
    var_9 = module_0.Not(var_7, **var_8)
    var_10 = 'some_value'
    var_11 = var_9.validate(var_10)
    var_12 = 'Should have raised validation error'
    var_13 = AssertionError(var_12)
    var_14 = 'negated'

def test_case_0():
    var_0 = 'value'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_one_of_validate_exactly_one_match. Retrieved 4/33 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = None
    var_2 = 'error'
    var_3 = 'test_value'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_raises_error_when_negated_field_matches. Retrieved 6/14 statements.
# Partially parsed test_validate_returns_value_when_negated_field_does_not_match. Retrieved 4/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'Must not match.'
    var_3 = ValueError(var_2)
    var_4 = 'some_value'
    var_5 = 'some_value'

def test_case_0():
    var_0 = False
    var_1 = 'Error occurred'
    var_2 = 'target_value'
    var_3 = 'target_value'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'some_value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_value'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_success_single_match. Retrieved 1/13 statements.
# Partially parsed test_validate_raises_no_match. Retrieved 1/11 statements.
# Partially parsed test_validate_raises_multiple_matches. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'unmatched'

def test_case_0():
    var_0 = 'value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nevermatch_validation_error_logic. Retrieved 4/7 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'Description'
    var_2 = 123
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'default'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.NeverMatch(**var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Field'
    var_9 = var_7.description
    assert var_9 == 'Description'
    var_10 = var_7.default
    assert var_10 == 123
    var_11 = var_7.allow_null
    assert var_11 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = 'any value'
    var_3 = var_1.validate(var_2)
    var_4 = str(var_2)
    var_5 = 'This never validates.'
    var_6 = bool('This never validates.' in var_4)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_not_constructor_passes_kwargs_to_super. Retrieved 3/4 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Negated'
    var_1 = 'Description'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'Not Field'
    var_4 = 'Not Description'
    var_5 = True
    var_6 = 'title'
    var_7 = 'description'
    var_8 = 'read_only'
    var_9 = {var_6: var_3, var_7: var_4, var_8: var_5}
    var_10 = module_1.Not(var_2, **var_9)
    var_11 = var_10.negated
    var_12 = bool(var_10.negated == var_2)
    assert var_12 is True
    var_13 = var_10.title
    assert var_13 == 'Not Field'
    var_14 = var_10.description
    assert var_14 == 'Not Description'
    var_15 = var_10.read_only
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default_val'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_if_then_else_constructor_defaults_clauses_to_any. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = 'test'
    var_4 = 'desc'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.IfThenElse(var_0, var_1, var_2, **var_7)
    var_9 = var_8.if_clause
    var_10 = bool(var_8.if_clause == var_0)
    assert var_10 is True
    var_11 = var_8.then_clause
    var_12 = bool(var_8.then_clause == var_1)
    assert var_12 is True
    var_13 = var_8.else_clause
    var_14 = bool(var_8.else_clause == var_2)
    assert var_14 is True
    var_15 = var_8.title
    assert var_15 == 'test'
    var_16 = var_8.description
    assert var_16 == 'desc'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.then_clause
    var_4 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_one_of_validate_multiple_matches. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #6
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
    var_6 = 'AllOf Field'
    var_7 = 'title'
    var_8 = {var_7: var_6}
    var_9 = module_1.AllOf(var_5, **var_8)
    var_10 = var_9.all_of
    var_11 = bool(var_9.all_of == [var_1, var_4])
    assert var_11 is True
    var_12 = var_9.title
    assert var_12 == 'AllOf Field'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = 'Should have raised AssertionError because allow_null is forbidden in AllOf kwargs'
    var_8 = AssertionError(var_7)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = 'Description'
    var_4 = 'description'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_2, **var_5)
    var_7 = var_6.description
    assert var_7 == 'Description'
    var_8 = var_6.allow_null
    assert var_8 is False



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.OneOf(var_1, **var_2)
    var_4 = var_3.title
    assert var_4 == ''
    var_5 = var_3.description
    assert var_5 == ''
    var_6 = var_3.allow_null
    assert var_6 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_one_of_init_raises_assertion_error_when_allow_null_is_provided. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_single_match. Retrieved 1/39 statements.
# Partially parsed test_validate_no_match. Retrieved 1/28 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 1/28 statements.


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'any_value'

def test_case_0():
    var_0 = 'any_value'



# Parsed testcases at query #10
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
    var_7 = 'title'
    var_8 = 'description'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_1.OneOf(var_4, **var_9)
    var_11 = var_10.one_of
    var_12 = bool(var_10.one_of == [var_1, var_3])
    assert var_12 is True
    var_13 = var_10.title
    assert var_13 == 'Union Field'
    var_14 = var_10.description
    assert var_14 == 'A test field'
    var_15 = var_10.allow_null
    assert var_15 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.OneOf(var_2, **var_5)
    var_7 = 'Should have raised an error because allow_null is not allowed in kwargs'
    var_8 = AssertionError(var_7)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.OneOf(var_2, **var_3)
    var_5 = var_4.title
    assert var_5 == ''
    var_6 = var_4.description
    assert var_6 == ''
    var_7 = var_4.allow_null
    assert var_7 is False



