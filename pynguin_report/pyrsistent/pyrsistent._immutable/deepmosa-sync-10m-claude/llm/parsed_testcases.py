####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_single_field. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 8/9 statements.
# Partially parsed test_immutable_set_no_changes. Retrieved 6/7 statements.
# Partially parsed test_immutable_set_invalid_member. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 8/9 statements.
# Partially parsed test_immutable_frozen_member_cannot_be_set. Retrieved 8/10 statements.
# Partially parsed test_immutable_single_member. Retrieved 6/7 statements.
# Partially parsed test_immutable_multiple_frozen_members. Retrieved 10/13 statements.
# Partially parsed test_immutable_set_with_frozen_and_normal_member. Retrieved 7/8 statements.


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
    var_6 = 5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is not a member'

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
    var_9 = 'Cannot set frozen members'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, z'
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

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x y z'
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
    var_0 = ''
    var_1 = 'Empty'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = var_2()
    var_4 = repr(var_3)
    assert var_4 == 'Empty()'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = module_0.immutable(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = var_1(var_2, var_3)
    var_5 = repr(var_4)
    var_6 = 'Immutable'
    var_7 = bool('Immutable' in var_5)
    assert var_7 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'CustomPoint'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 5
    var_4 = 10
    var_5 = var_2(var_3, var_4)
    var_6 = repr(var_5)
    var_7 = 'CustomPoint'
    var_8 = bool('CustomPoint' in var_6)
    assert var_8 is True
    var_9 = repr(var_5)
    var_10 = 'ImmutableBase'
    var_11 = bool('ImmutableBase' not in var_9)
    assert var_11 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 'Value'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 42
    var_4 = var_2(var_3)
    var_5 = var_4.val
    assert var_5 == 42
    var_6 = 100

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, id_, y, uuid_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 'id1'
    var_5 = 2
    var_6 = 'uuid1'
    var_7 = var_2(var_3, var_4, var_5, var_6)
    var_8 = 10
    var_9 = 'id2'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Cannot set frozen members'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, id_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 'id1'
    var_5 = var_2(var_3, var_4)
    var_6 = 5

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = var_3(var_4, var_5)
    var_7 = var_6.x
    assert var_7 == 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_method_multiple_fields. Retrieved 9/10 statements.
# Partially parsed test_immutable_set_method_no_changes. Retrieved 6/7 statements.
# Partially parsed test_immutable_set_nonexistent_member. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 8/9 statements.
# Partially parsed test_immutable_frozen_member_cannot_be_modified. Retrieved 8/10 statements.
# Partially parsed test_immutable_multiple_frozen_members. Retrieved 10/11 statements.
# Partially parsed test_immutable_multiple_frozen_members_error. Retrieved 9/11 statements.
# Partially parsed test_immutable_single_member. Retrieved 6/7 statements.
# Partially parsed test_immutable_as_base_class. Retrieved 3/9 statements.
# Partially parsed test_immutable_as_base_class_validation. Retrieved 3/10 statements.
# Partially parsed test_immutable_as_base_class_set_with_validation. Retrieved 4/11 statements.
# Partially parsed test_immutable_as_base_class_set_invalid. Retrieved 4/12 statements.
# Partially parsed test_immutable_namedtuple_behavior. Retrieved 6/7 statements.


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
    var_0 = 'x, y, z'
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

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x y z'
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
    var_6 = 3
    var_7 = var_5.x
    assert var_7 == 1
    var_8 = var_5.y
    assert var_8 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y, z'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(var_3, var_4, var_5)
    var_7 = 10
    var_8 = 30

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
    var_8 = 'is not a member'

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
    var_9 = 'Cannot set frozen members'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b_, c, d_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = var_2(var_3, var_4, var_5, var_6)
    var_8 = 10
    var_9 = 30

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b_, c, d_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = var_2(var_3, var_4, var_5, var_6)
    var_8 = 20
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Cannot set frozen members'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = repr(var_5)
    var_7 = 'Point'
    var_8 = bool('Point' in var_6)
    assert var_8 is True
    var_9 = repr(var_5)
    var_10 = 'x=1'
    var_11 = bool('x=1' in var_9)
    assert var_11 is True
    var_12 = repr(var_5)
    var_13 = 'y=2'
    var_14 = bool('y=2' in var_12)
    assert var_14 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'Empty'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = var_2()
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'Single'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 42
    var_4 = var_2(var_3)
    var_5 = var_4.value
    assert var_5 == 42
    var_6 = 100

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = tuple()
    var_1 = -1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Coordinates must be positive!'

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Coordinates must be positive!'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = var_3(var_4, var_5)
    var_7 = var_6.x
    assert var_7 == 1

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
    var_6 = var_2(var_3, var_4)
    var_7 = 3
    var_8 = var_2(var_4, var_7)
    var_9 = bool(var_5 == var_6)
    assert var_9 is True
    var_10 = bool(var_5 != var_8)
    assert var_10 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_2(var_3, var_4)
    var_7 = hash(var_5)
    var_8 = hash(var_6)
    var_9 = bool(var_7 == var_8)
    assert var_9 is True
    var_10 = {var_5, var_6}
    var_11 = len(var_10)
    assert var_11 == 1



