####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_returns_value_when_negated_field_has_error. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_validation_error_when_negated_field_validates. Retrieved 3/8 statements.


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



# Parsed testcases at query #2
#--------------------------




import typesystem.composites as module_1
import typesystem.fields as module_0


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
    var_8 = var_4.read_only
    assert var_8 is False
    var_9 = var_4.title
    assert var_9 == ''
    var_10 = var_4.description
    assert var_10 == ''


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.OneOf(var_1, **var_2)
    var_4 = var_3.one_of
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_3.one_of[0]
    var_7 = bool(var_3.one_of[0] is var_0)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.OneOf(var_3, **var_4)
    var_6 = var_5.one_of
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = var_5.one_of
    var_9 = bool(var_5.one_of == [var_0, var_1, var_2])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------





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


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'Title'
    var_3 = 'Description'
    var_4 = True
    var_5 = 'title'
    var_6 = 'description'
    var_7 = 'read_only'
    var_8 = {var_5: var_2, var_6: var_3, var_7: var_4}
    var_9 = module_1.Not(var_1, **var_8)
    var_10 = var_9.negated
    var_11 = bool(var_9.negated is var_1)
    assert var_11 is True
    var_12 = var_9.allow_null
    assert var_12 is False
    var_13 = var_9.read_only
    assert var_13 is True
    var_14 = var_9.title
    assert var_14 == 'Title'
    var_15 = var_9.description
    assert var_15 == 'Description'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.Not(var_1, **var_4)
    var_6 = bool(False)
    assert var_6 is True


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



# Parsed testcases at query #4
#--------------------------





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


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.AllOf(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0


def test_case_0():
    var_0 = 'Title1'
    var_1 = 'Desc1'
    var_2 = True
    var_3 = module_0.Field(title=var_0, description=var_1, read_only=var_2)
    var_4 = 'Title2'
    var_5 = 'Desc2'
    var_6 = False
    var_7 = module_0.Field(title=var_4, description=var_5, read_only=var_6)
    var_8 = [var_3, var_7]
    var_9 = 'AllTitle'
    var_10 = 'AllDesc'
    var_11 = 'title'
    var_12 = 'description'
    var_13 = 'read_only'
    var_14 = {var_11: var_9, var_12: var_10, var_13: var_2}
    var_15 = module_1.AllOf(var_8, **var_14)
    var_16 = var_15.title
    assert var_16 == 'AllTitle'
    var_17 = var_15.description
    assert var_17 == 'AllDesc'
    var_18 = var_15.read_only
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.AllOf(var_1, **var_2)
    var_4 = 'default'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_3.allow_null
    assert var_7 is False
    var_8 = var_3.read_only
    assert var_8 is False
    var_9 = var_3.title
    assert var_9 == ''
    var_10 = var_3.description
    assert var_10 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_if_then_else_constructor_with_only_if_clause. Retrieved 4/8 statements.
# Partially parsed test_if_then_else_constructor_with_then_clause_only. Retrieved 4/7 statements.
# Partially parsed test_if_then_else_constructor_with_else_clause_only. Retrieved 4/7 statements.



def test_case_0():
    var_0 = module_0.Any()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause is var_0)
    assert var_4 is True
    var_5 = var_2.then_clause
    var_6 = var_2.else_clause


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




import typesystem.composites as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False


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


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.read_only
    assert var_2 is False


def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.errors
    var_3 = bool(var_1.errors == {'never': 'This never validates.'})
    assert var_3 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_one_match. Retrieved 10/12 statements.
# Partially parsed test_validate_no_match. Retrieved 10/13 statements.
# Partially parsed test_validate_multiple_matches. Retrieved 10/15 statements.
# Partially parsed test_validate_with_null_not_allowed. Retrieved 10/13 statements.
# Partially parsed test_validate_single_field_list. Retrieved 8/9 statements.
# Partially parsed test_validate_candidate_preserved. Retrieved 11/13 statements.
# Partially parsed test_validate_error_from_child. Retrieved 11/13 statements.


import typesystem.fields as module_0


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
    var_9 = 5.5
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True


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
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = None
    var_3 = 'error'
    var_4 = (var_2, var_3)
    var_5 = module_0.Field()
    var_6 = (var_2, var_3)
    var_7 = [var_0, var_5]
    var_8 = {}
    var_9 = module_1.OneOf(var_7, **var_8)
    var_10 = 21
    var_11 = var_9.validate(var_10)
    assert var_11 == 42


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'child_error'
    var_3 = (var_1, var_2)
    var_4 = module_0.Field()
    var_5 = 'error'
    var_6 = (var_1, var_5)
    var_7 = [var_0, var_4]
    var_8 = {}
    var_9 = module_1.OneOf(var_7, **var_8)
    var_10 = 'test'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_if_then_else_constructor_with_only_if_clause. Retrieved 4/8 statements.
# Partially parsed test_if_then_else_constructor_with_then_clause. Retrieved 4/7 statements.
# Partially parsed test_if_then_else_constructor_with_else_clause. Retrieved 4/7 statements.



def test_case_0():
    var_0 = module_0.Any()
    var_1 = {}
    var_2 = module_1.IfThenElse(var_0, **var_1)
    var_3 = var_2.if_clause
    var_4 = bool(var_2.if_clause is var_0)
    assert var_4 is True
    var_5 = var_2.then_clause
    var_6 = var_2.else_clause


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


def test_case_0():
    var_0 = module_0.Any()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.IfThenElse(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Any()
    var_1 = 'Title'
    var_2 = 'Description'
    var_3 = True
    var_4 = 'title'
    var_5 = 'description'
    var_6 = 'read_only'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_1.IfThenElse(var_0, **var_7)
    var_9 = var_8.title
    assert var_9 == 'Title'
    var_10 = var_8.description
    assert var_10 == 'Description'
    var_11 = var_8.read_only
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.composites as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.allow_null
    assert var_2 is False


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


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False


def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.NeverMatch(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.OneOf(var_2, **var_5)
    var_7 = var_6.allow_null
    assert var_7 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 'default_value'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = var_5.default
    assert var_6 == 'default_value'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.OneOf(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.OneOf(var_1, **var_2)
    var_4 = var_3.one_of
    var_5 = bool(var_3.one_of == [var_0])
    assert var_5 is True

import typesystem.composites as module_0


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.OneOf(var_0, **var_1)
    var_3 = var_2.one_of
    var_4 = bool(var_2.one_of == [])
    assert var_4 is True

import typesystem.fields as module_0


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





def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = var_4.all_of
    var_6 = bool(var_4.all_of == [var_0, var_1])
    assert var_6 is True


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


def test_case_0():
    var_0 = []
    var_1 = 'Test'
    var_2 = 'Description'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.AllOf(var_0, **var_5)
    var_7 = var_6.title
    assert var_7 == 'Test'
    var_8 = var_6.description
    assert var_8 == 'Description'

import typesystem.fields as module_0


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.AllOf(var_1, **var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = 'Field1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field2'
    var_3 = module_0.Field(description=var_2)
    var_4 = module_0.Field()
    var_5 = [var_1, var_3, var_4]
    var_6 = {}
    var_7 = module_1.AllOf(var_5, **var_6)
    var_8 = var_7.all_of
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_7.all_of[0]
    var_11 = bool(var_7.all_of[0] == var_1)
    assert var_11 is True
    var_12 = var_7.all_of[1]
    var_13 = bool(var_7.all_of[1] == var_3)
    assert var_13 is True
    var_14 = var_7.all_of[2]
    var_15 = bool(var_7.all_of[2] == var_4)
    assert var_15 is True



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = var_2.negated
    var_4 = bool(var_2.negated is var_0)
    assert var_4 is True
    var_5 = var_2.allow_null
    assert var_5 is False


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Title'
    var_2 = 'Description'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_1.Not(var_0, **var_5)
    var_7 = var_6.negated
    var_8 = bool(var_6.negated is var_0)
    assert var_8 is True
    var_9 = var_6.title
    assert var_9 == 'Title'
    var_10 = var_6.description
    assert var_10 == 'Description'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Not(var_0, **var_1)
    var_3 = var_2.read_only
    assert var_3 is False


def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_1.Not(var_0, **var_3)
    var_5 = var_4.read_only
    assert var_5 is True



