####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 8/9 statements.
# Partially parsed test_immutable_set_no_change. Retrieved 6/7 statements.
# Partially parsed test_immutable_invalid_field_error. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 9/12 statements.
# Partially parsed test_immutable_multiple_frozen_members. Retrieved 12/15 statements.
# Partially parsed test_immutable_inheritance. Retrieved 5/14 statements.
# Partially parsed test_immutable_no_members. Retrieved 4/5 statements.
# Partially parsed test_immutable_single_member. Retrieved 6/7 statements.
# Partially parsed test_immutable_members_as_list. Retrieved 9/10 statements.
# Partially parsed test_immutable_members_with_commas_and_spaces. Retrieved 8/9 statements.


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2
    var_8 = str(var_5)
    assert var_8 == 'Point(x=1, y=2)'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = var_5.x
    assert var_7 == 1
    var_8 = var_5.y
    assert var_8 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = 4

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = bool(False)
    assert var_7 is True
    var_8 = "'z' is not a member"

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = var_6.id_
    assert var_7 == 17
    var_8 = 3
    var_9 = 18
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Cannot set frozen members id_'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b_, c, d_'
    var_1 = 'Thing'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = var_2(var_3, var_4, var_5, var_6)
    var_8 = var_7.b_
    assert var_8 == 2
    var_9 = var_7.d_
    assert var_9 == 4
    var_10 = 10
    var_11 = 30
    var_12 = 20
    var_13 = 40
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'Cannot set frozen members'
    var_16 = 'b_'
    var_17 = 'd_'

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = -3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Coordinates must be positive!'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = module_0.immutable(name=var_0)
    var_2 = var_1()
    var_3 = str(var_2)
    assert var_3 == 'Empty()'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'Single'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 5
    var_4 = var_2(var_3)
    var_5 = var_4.x
    assert var_5 == 5
    var_6 = 10

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b'
    var_1 = 'MyThing'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = str(var_5)
    assert var_6 == 'MyThing(a=1, b=2)'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = 'Point'
    var_4 = module_0.immutable(var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = var_4(var_5, var_6)
    var_8 = var_7.x
    assert var_8 == 1
    var_9 = var_7.y
    assert var_9 == 2
    var_10 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x,  y   ,z'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(var_3, var_4, var_5)
    var_7 = var_6.x
    assert var_7 == 1
    var_8 = var_6.y
    assert var_8 == 2
    var_9 = var_6.z
    assert var_9 == 3
    var_10 = 20



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_updates_field. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_returns_new_instance. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_no_changes_returns_self. Retrieved 6/7 statements.
# Partially parsed test_immutable_inheritance_with_custom_new. Retrieved 3/9 statements.
# Partially parsed test_immutable_inheritance_set_respects_custom_new. Retrieved 4/11 statements.
# Partially parsed test_immutable_frozen_member_cannot_be_set. Retrieved 8/10 statements.
# Partially parsed test_immutable_frozen_member_allows_other_updates. Retrieved 8/9 statements.
# Partially parsed test_immutable_set_invalid_member_raises_error. Retrieved 7/9 statements.
# Partially parsed test_immutable_set_on_no_members_returns_self. Retrieved 3/4 statements.
# Partially parsed test_immutable_multiple_frozen_members. Retrieved 9/11 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 8/9 statements.


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = repr(var_5)
    assert var_6 == 'Point(x=1, y=2)'

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 18
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = bool(False)
    assert var_7 is True
    var_8 = "'z' is not a member"

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = 'Point'
    var_4 = module_0.immutable(var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = var_4(var_5, var_6)
    var_8 = var_7.x
    assert var_8 == 1
    var_9 = var_7.y
    assert var_9 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = module_0.immutable(name=var_0)
    var_2 = var_1()
    var_3 = repr(var_2)
    assert var_3 == 'Empty()'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = module_0.immutable(name=var_0)
    var_2 = var_1()

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a_, b, c_'
    var_1 = 'Thing'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(a_=var_3, b=var_4, c_=var_5)
    var_7 = 4
    var_8 = 5
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Cannot set frozen members a_, c_'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #2
#--------------------------




import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'InvalidClass'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_91_evaluates_to_false. Retrieved 7/8 statements.


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = var_5.x
    assert var_7 == 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_immutable_with_empty_members_and_name_immutable. Retrieved 6/7 statements.
# Partially parsed test_immutable_with_single_member_and_no_frozen. Retrieved 6/7 statements.
# Partially parsed test_immutable_with_multiple_members_and_frozen. Retrieved 9/12 statements.
# Partially parsed test_immutable_inheritance_and_custom_new. Retrieved 5/14 statements.
# Partially parsed test_immutable_with_invalid_member_in_set. Retrieved 7/9 statements.
# Partially parsed test_immutable_with_empty_kwargs_in_set. Retrieved 6/7 statements.
# Partially parsed test_immutable_with_frozen_member_and_multiple_fields_to_modify. Retrieved 9/11 statements.
# Partially parsed test_immutable_with_no_frozen_members. Retrieved 8/9 statements.
# Partially parsed test_immutable_verbose_false_no_output. Retrieved 4/5 statements.


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'Immutable'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)
    var_4 = var_3()
    var_5 = var_4._fields
    var_6 = bool(var_4._fields == ())
    assert var_6 is True
    var_7 = repr(var_4)
    assert var_7 == 'Immutable()'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 5
    var_4 = var_2(var_3)
    var_5 = 10
    var_6 = var_4.x
    assert var_6 == 5

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 3
    var_8 = 18
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = -3
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = bool(False)
    assert var_7 is True
    var_8 = "'z' is not a member"

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = repr(var_5)
    assert var_6 == 'Point(x=1, y=2)'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 3
    var_8 = 18
    var_9 = bool(False)
    assert var_9 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = 4

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_immutable_set_updates_field. Retrieved 7/9 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 8/9 statements.
# Partially parsed test_immutable_set_no_change. Retrieved 6/7 statements.
# Partially parsed test_immutable_set_invalid_field. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 9/12 statements.
# Partially parsed test_immutable_inheritance. Retrieved 5/14 statements.
# Partially parsed test_immutable_no_members. Retrieved 4/5 statements.
# Partially parsed test_immutable_set_frozen_multiple. Retrieved 12/15 statements.


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2
    var_8 = repr(var_5)
    assert var_8 == 'Point(x=1, y=2)'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = 4

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = bool(False)
    assert var_7 is True
    var_8 = "'z' is not a member"

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = var_6.id_
    assert var_7 == 17
    var_8 = 3
    var_9 = 18
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Cannot set frozen members id_'

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = -3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Coordinates must be positive!'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = 'Point'
    var_4 = module_0.immutable(var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = var_4(var_5, var_6)
    var_8 = var_7.x
    assert var_8 == 1
    var_9 = var_7.y
    assert var_9 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 1
    var_7 = var_5.y
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = module_0.immutable(name=var_0)
    var_2 = var_1()
    var_3 = repr(var_2)
    assert var_3 == 'Empty()'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'MyPoint'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = repr(var_5)
    assert var_6 == 'MyPoint(x=1, y=2)'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b_, c, d_'
    var_1 = 'Obj'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = var_2(var_3, var_4, var_5, var_6)
    var_8 = var_7.b_
    assert var_8 == 2
    var_9 = var_7.d_
    assert var_9 == 4
    var_10 = 5
    var_11 = 6
    var_12 = 7
    var_13 = 8
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'Cannot set frozen members'
    var_16 = 'b_'
    var_17 = 'd_'



