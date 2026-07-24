####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_if_then_else_constructor_with_only_if_clause. Retrieved 4/8 statements.
# Partially parsed test_if_then_else_constructor_with_then_clause. Retrieved 4/7 statements.
# Partially parsed test_if_then_else_constructor_with_else_clause. Retrieved 4/7 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause is var_0)
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
    var_5 = bool(var_3.if_clause is var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = bool(var_3.then_clause is var_1)
    assert var_7 is True
    var_8 = var_3.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, else_clause=var_1, **var_2)
    var_4 = var_3.if_clause
    var_5 = bool(var_3.if_clause is var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = var_3.else_clause
    var_8 = bool(var_3.else_clause is var_1)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_0.Any()
    var_2 = module_0.Any()
    var_3 = {}
    var_4 = module_1.IfThenElse(var_0, var_1, var_2, **var_3)
    var_5 = var_4.if_clause
    var_6 = bool(var_4.if_clause is var_0)
    assert var_6 is True
    var_7 = var_4.then_clause
    var_8 = bool(var_4.then_clause is var_1)
    assert var_8 is True
    var_9 = var_4.else_clause
    var_10 = bool(var_4.else_clause is var_2)
    assert var_10 is True

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
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.all_of
    var_5 = bool(var_3.all_of == [var_0])
    assert var_5 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = var_2.all_of
    var_4 = bool(var_2.all_of == [])
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.allow_null
    assert var_4 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = 'Test description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.AllOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test'
    var_9 = var_7.description
    assert var_9 == 'Test description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'default'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = var_5.default
    assert var_6 == 'default'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'callable'
    var_3 = lambda : var_2
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_1, **var_5)
    var_7 = var_6.default
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True



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
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.all_of
    var_5 = bool(var_3.all_of == [var_0])
    assert var_5 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = var_2.all_of
    var_4 = bool(var_2.all_of == [])
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = var_6.allow_null
    assert var_7 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = 'Description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.AllOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test'
    var_9 = var_7.description
    assert var_9 == 'Description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.allow_null
    assert var_4 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = var_0 | var_1
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.AllOf(var_3, **var_4)
    var_6 = var_5.all_of
    var_7 = bool(var_5.all_of == [var_2])
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = [var_1]
    var_3 = 'read_only'
    var_4 = {var_3: var_0}
    var_5 = module_1.AllOf(var_2, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'default'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = var_5.default
    assert var_6 == 'default'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'callable'
    var_3 = lambda : var_2
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_1, **var_5)
    var_7 = var_6.default
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_validation_error_when_negated_field_has_no_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = None
    var_1 = 'error'
    var_2 = 'test_value'

def test_case_0():
    var_0 = 'valid'
    var_1 = None
    var_2 = 'test_value'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = var_3.negated
    var_5 = bool(var_3.negated is var_1)
    assert var_5 is True
    var_6 = var_3.allow_null
    assert var_6 is False
    var_7 = var_3.read_only
    assert var_7 is False
    var_8 = var_3.title
    assert var_8 == ''
    var_9 = var_3.description
    assert var_9 == ''

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'Test Title'
    var_3 = 'Test Description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.Not(var_1, **var_6)
    var_8 = var_7.negated
    var_9 = bool(var_7.negated is var_1)
    assert var_9 is True
    var_10 = var_7.title
    assert var_10 == 'Test Title'
    var_11 = var_7.description
    assert var_11 == 'Test Description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.Not(var_1, **var_4)
    var_6 = var_5.negated
    var_7 = bool(var_5.negated is var_1)
    assert var_7 is True
    var_8 = var_5.read_only
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.Not(var_1, **var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = var_1 | var_3
    var_5 = {}
    var_6 = module_1.Not(var_4, **var_5)
    var_7 = var_6.negated
    var_8 = bool(var_6.negated is var_4)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = {}
    var_5 = module_1.Not(var_3, **var_4)
    var_6 = var_5.negated
    var_7 = bool(var_5.negated is var_3)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_nevermatch_constructor_with_default. Retrieved 2/4 statements.
# Partially parsed test_nevermatch_constructor_with_callable_default. Retrieved 3/5 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False
    var_3 = var_1.read_only
    assert var_3 is False
    var_4 = var_1.title
    assert var_4 == ''
    var_5 = var_1.description
    assert var_5 == ''

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

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'callable_default'
    var_1 = lambda : var_0
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.NeverMatch(**var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_oneof_constructor_with_default. Retrieved 4/6 statements.
# Partially parsed test_oneof_constructor_with_callable_default. Retrieved 5/7 statements.


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
    var_7 = var_4.allow_null
    assert var_7 is False

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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'default_value'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'callable_default'
    var_3 = lambda : var_2
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_1.OneOf(var_1, **var_5)

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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_matches_exactly_one. Retrieved 10/12 statements.
# Partially parsed test_validate_no_match. Retrieved 10/13 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 10/15 statements.
# Partially parsed test_validate_with_null_not_allowed. Retrieved 9/11 statements.
# Partially parsed test_validate_single_field_list. Retrieved 8/9 statements.
# Partially parsed test_validate_returns_validated_value. Retrieved 12/14 statements.
# Partially parsed test_validate_nested_one_of. Retrieved 15/18 statements.


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
    var_7 = {}
    var_8 = module_1.OneOf(var_6, **var_7)
    var_9 = 5
    var_10 = var_8.validate(var_9)
    assert var_10 == 5

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
    var_7 = {}
    var_8 = module_1.OneOf(var_6, **var_7)
    var_9 = 5.0
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_7 = {}
    var_8 = module_1.OneOf(var_6, **var_7)
    var_9 = 5.0
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_7 = {}
    var_8 = module_1.OneOf(var_6, **var_7)
    var_9 = var_8.validate(var_1)
    assert var_9 is None

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.OneOf(var_0, **var_1)
    var_3 = 'anything'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'TEST'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = None
    var_3 = 'error'
    var_4 = (var_2, var_3)
    var_5 = module_0.Field()
    var_6 = ' processed'
    var_7 = (var_2, var_3)
    var_8 = [var_0, var_5]
    var_9 = {}
    var_10 = module_1.OneOf(var_8, **var_9)
    var_11 = 3
    var_12 = var_10.validate(var_11)
    assert var_12 == 6

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'inner1'
    var_2 = None
    var_3 = 'error'
    var_4 = (var_2, var_3)
    var_5 = module_0.Field()
    var_6 = 'inner2'
    var_7 = (var_2, var_3)
    var_8 = [var_0, var_5]
    var_9 = {}
    var_10 = module_1.OneOf(var_8, **var_9)
    var_11 = module_0.Field()
    var_12 = (var_2, var_3)
    var_13 = [var_10, var_11]
    var_14 = {}
    var_15 = module_1.OneOf(var_13, **var_14)
    var_16 = var_15.validate(var_1)
    assert var_16 == 'inner1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_match_count_equals_one_returns_candidate. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 1/8 statements.
# Partially parsed test_validate_raises_error_when_negated_field_has_no_error. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'test_value'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_match_count_equals_one_returns_candidate. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_validation_error_when_negated_field_has_no_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = None
    var_1 = 'some error'
    var_2 = 'test_value'

def test_case_0():
    var_0 = 'valid'
    var_1 = None
    var_2 = 'test_value'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #13
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
    var_7 = var_4.allow_null
    assert var_7 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.all_of
    var_5 = bool(var_3.all_of == [var_0])
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.AllOf(var_3, **var_4)
    var_6 = var_5.all_of
    var_7 = bool(var_5.all_of == [var_0, var_1, var_2])
    assert var_7 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = var_2.all_of
    var_4 = bool(var_2.all_of == [])
    assert var_4 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.AllOf(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'desc'
    var_2 = True
    var_3 = module_0.Field(title=var_0, description=var_1, read_only=var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = var_6.title
    assert var_7 == ''
    var_8 = var_6.description
    assert var_8 == ''
    var_9 = var_6.read_only
    assert var_9 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = var_0 | var_1
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.AllOf(var_3, **var_4)
    var_6 = var_5.all_of
    var_7 = bool(var_5.all_of == [var_2])
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_validation_error_when_negated_field_has_no_error. Retrieved 4/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'some error'
    var_2 = 'test_value'

def test_case_0():
    var_0 = 'valid'
    var_1 = None
    var_2 = 'Must not match.'
    var_3 = [var_2]
    var_4 = 'test_value'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_match_count_equals_one_returns_candidate. Retrieved 1/8 statements.
# Partially parsed test_match_count_equals_one_with_multiple_fields_only_one_matches. Retrieved 1/12 statements.
# Partially parsed test_match_count_equals_one_with_three_fields_only_one_matches. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #16
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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_0.Any()
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.OneOf(var_3, **var_4)
    var_6 = 'hello'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_1.OneOf(var_6, **var_7)
    var_9 = True
    var_10 = var_8.validate(var_9)
    assert var_10 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.OneOf(var_0, **var_1)
    var_3 = 'anything'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_multiple_matches. Retrieved 1/11 statements.
# Partially parsed test_validate_with_nested_fields. Retrieved 6/11 statements.


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
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123

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
    var_0 = 'anything'
    var_1 = bool(False)
    assert var_1 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

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
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nevermatch_constructor_with_default. Retrieved 2/4 statements.
# Partially parsed test_nevermatch_constructor_with_callable_default. Retrieved 3/5 statements.


import typesystem.composites as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'Title'
    var_1 = 'Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.NeverMatch(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Title'
    var_7 = var_5.description
    assert var_7 == 'Description'

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
    var_0 = 'default'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)

import typesystem.composites as module_0

def test_case_0():
    var_0 = 'callable'
    var_1 = lambda : var_0
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.NeverMatch(**var_3)

import typesystem.composites as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_if_then_else_constructor_with_only_if_clause. Retrieved 4/8 statements.
# Partially parsed test_if_then_else_constructor_with_then_clause. Retrieved 4/7 statements.
# Partially parsed test_if_then_else_constructor_with_else_clause. Retrieved 4/7 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause is var_0)
    assert var_4 is True
    var_5 = var_2.then_clause
    var_6 = var_2.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, var_1, **var_2)
    var_4 = var_3.if_clause
    var_5 = bool(var_3.if_clause is var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = bool(var_3.then_clause is var_1)
    assert var_7 is True
    var_8 = var_3.else_clause

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_1.IfThenElse(var_0, else_clause=var_1, **var_2)
    var_4 = var_3.if_clause
    var_5 = bool(var_3.if_clause is var_0)
    assert var_5 is True
    var_6 = var_3.then_clause
    var_7 = var_3.else_clause
    var_8 = bool(var_3.else_clause is var_1)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = {}
    var_4 = module_1.IfThenElse(var_0, var_1, var_2, **var_3)
    var_5 = var_4.if_clause
    var_6 = bool(var_4.if_clause is var_0)
    assert var_6 is True
    var_7 = var_4.then_clause
    var_8 = bool(var_4.then_clause is var_1)
    assert var_8 is True
    var_9 = var_4.else_clause
    var_10 = bool(var_4.else_clause is var_2)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'allow_null'



# Parsed testcases at query #4
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
    var_7 = var_4.allow_null
    assert var_7 is False

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

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.OneOf(var_0, **var_1)
    var_3 = var_2.one_of
    var_4 = bool(var_2.one_of == [])
    assert var_4 is True

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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.OneOf(var_3, **var_4)
    var_6 = var_5.one_of
    var_7 = bool(var_5.one_of == [var_0, var_1, var_2])
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_true. Retrieved 1/18 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = var_2.negated
    var_4 = bool(var_2.negated is var_0)
    assert var_4 is True
    var_5 = var_2.allow_null
    assert var_5 is False
    var_6 = var_2.read_only
    assert var_6 is False
    var_7 = var_2.title
    assert var_7 == ''
    var_8 = var_2.description
    assert var_8 == ''

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Title'
    var_2 = 'Description'
    var_3 = True
    var_4 = 'title'
    var_5 = 'description'
    var_6 = 'read_only'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_1.Not(var_0, **var_7)
    var_9 = var_8.negated
    var_10 = bool(var_8.negated is var_0)
    assert var_10 is True
    var_11 = var_8.allow_null
    assert var_11 is False
    var_12 = var_8.read_only
    assert var_12 is True
    var_13 = var_8.title
    assert var_13 == 'Title'
    var_14 = var_8.description
    assert var_14 == 'Description'

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

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = var_3.negated
    var_5 = bool(var_3.negated is var_1)
    assert var_5 is True
    var_6 = var_3.allow_null
    assert var_6 is False



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
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
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = True
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_0.Any()
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.OneOf(var_3, **var_4)
    var_6 = 456
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.Boolean(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_1.OneOf(var_9, **var_10)
    var_12 = True
    var_13 = var_11.validate(var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.Boolean(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_1.OneOf(var_9, **var_10)
    var_12 = 789
    var_13 = var_11.validate(var_12)
    assert var_13 == 789

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.OneOf(var_0, **var_1)
    var_3 = 'anything'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
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

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = var_2.all_of
    var_4 = bool(var_2.all_of == [])
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.all_of
    var_5 = bool(var_3.all_of == [var_0])
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'test description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.AllOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'test'
    var_9 = var_7.description
    assert var_9 == 'test description'
    var_10 = var_7.allow_null
    assert var_10 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.read_only
    assert var_4 is False
    var_5 = 'default'
    var_6 = hasattr(var_3, var_5)
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'field2'
    var_3 = module_0.Field(description=var_2)
    var_4 = True
    var_5 = module_0.Field(allow_null=var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_1.AllOf(var_6, **var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_8.all_of[0].title
    assert var_11 == 'field1'
    var_12 = var_8.all_of[1].description
    assert var_12 == 'field2'
    var_13 = var_8.all_of[2].allow_null
    assert var_13 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_allof_constructor_with_default. Retrieved 4/6 statements.
# Partially parsed test_allof_constructor_with_callable_default. Retrieved 5/7 statements.


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
    var_7 = var_4.allow_null
    assert var_7 is False

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = var_3.all_of
    var_5 = bool(var_3.all_of == [var_0])
    assert var_5 is True

import typesystem.composites as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.AllOf(var_0, **var_1)
    var_3 = var_2.all_of
    var_4 = bool(var_2.all_of == [])
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'Test'
    var_3 = 'Test description'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.AllOf(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test'
    var_9 = var_7.description
    assert var_9 == 'Test description'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'default'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'callable'
    var_3 = lambda : var_2
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_1.AllOf(var_1, **var_5)

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True



