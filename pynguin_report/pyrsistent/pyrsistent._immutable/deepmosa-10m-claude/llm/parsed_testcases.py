####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 8/9 statements.
# Partially parsed test_immutable_set_no_changes. Retrieved 6/7 statements.
# Partially parsed test_immutable_set_invalid_member. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 8/9 statements.
# Partially parsed test_immutable_frozen_member_cannot_set. Retrieved 8/10 statements.
# Partially parsed test_immutable_set_with_frozen_and_regular. Retrieved 9/10 statements.
# Partially parsed test_immutable_frozen_member_with_multiple. Retrieved 9/12 statements.
# Partially parsed test_immutable_immutability. Retrieved 6/8 statements.


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
    var_6 = 5
    var_7 = 10

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
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 5
    var_4 = 10
    var_5 = var_2(var_3, var_4)
    var_6 = var_5.x
    assert var_6 == 5
    var_7 = var_5.y
    assert var_7 == 10

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
    var_0 = 'value'
    var_1 = 'Single'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 42
    var_4 = var_2(var_3)
    var_5 = var_4.value
    assert var_5 == 42

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a, b, id_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 100
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 10
    var_8 = 20

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'a_, b_, c'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(var_3, var_4, var_5)
    var_7 = 5
    var_8 = 10
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
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]
    assert var_7 == 2
    var_8 = len(var_5)
    assert var_8 == 2

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
    var_8 = var_2(var_3, var_7)
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
    var_6 = bool(False)
    assert var_6 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/8 statements.
# Partially parsed test_immutable_empty_set. Retrieved 6/7 statements.
# Partially parsed test_immutable_invalid_member. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_member. Retrieved 8/9 statements.
# Partially parsed test_immutable_frozen_member_cannot_set. Retrieved 8/10 statements.
# Partially parsed test_immutable_multiple_frozen_members. Retrieved 10/11 statements.
# Partially parsed test_immutable_frozen_and_normal_mixed. Retrieved 9/11 statements.
# Partially parsed test_immutable_single_member. Retrieved 6/7 statements.
# Partially parsed test_immutable_set_multiple_fields. Retrieved 10/11 statements.
# Partially parsed test_immutable_original_unchanged_after_set. Retrieved 8/9 statements.
# Partially parsed test_immutable_inheritance. Retrieved 3/9 statements.
# Partially parsed test_immutable_inheritance_set_valid. Retrieved 4/11 statements.
# Partially parsed test_immutable_inheritance_set_invalid. Retrieved 4/12 statements.


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
    var_0 = 'a, b_, c'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(var_3, var_4, var_5)
    var_7 = 10
    var_8 = 20
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Cannot set frozen members'

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
    var_0 = 'value'
    var_1 = 'Single'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 42
    var_4 = var_2(var_3)
    var_5 = var_4.value
    assert var_5 == 42
    var_6 = 100

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'Empty'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = var_2()
    var_4 = repr(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

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
    var_8 = 20
    var_9 = 30

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 5
    var_7 = 6
    var_8 = var_5.x
    assert var_8 == 1
    var_9 = var_5.y
    assert var_9 == 2

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2

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
    var_5 = 'Coordinates must be positive'

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]
    assert var_7 == 2

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_2(var_3, var_4)
    var_7 = var_2(var_4, var_3)
    var_8 = bool(var_5 == var_6)
    assert var_8 is True
    var_9 = bool(var_5 != var_7)
    assert var_9 is True

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = var_2(var_3, var_4)
    var_7 = {var_5, var_6}
    var_8 = len(var_7)
    assert var_8 == 1

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
    var_1 = 'Point'
    var_2 = False
    var_3 = module_0.immutable(var_0, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = var_3(var_4, var_5)
    var_7 = var_6.x
    assert var_7 == 1



