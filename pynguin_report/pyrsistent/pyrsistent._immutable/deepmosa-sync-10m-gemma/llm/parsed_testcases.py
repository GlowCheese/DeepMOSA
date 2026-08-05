####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_basic_functionality. Retrieved 8/10 statements.
# Partially parsed test_immutable_no_members. Retrieved 5/6 statements.
# Partially parsed test_immutable_frozen_members. Retrieved 9/12 statements.
# Partially parsed test_immutable_invalid_member_setting. Retrieved 7/9 statements.
# Partially parsed test_immutable_inheritance. Retrieved 4/12 statements.


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
    var_9 = repr(var_5)
    assert var_9 == 'Point(x=1, y=2)'

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
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)
    var_7 = 3
    var_8 = 18
    var_9 = "Cannot set frozen members 'id_'"

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 5
    var_7 = "'z' is not a member"

def test_case_0():
    var_0 = tuple()
    var_1 = 1
    var_2 = 2
    var_3 = 3

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x,y,z'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = var_2(var_3, var_4, var_5)
    var_7 = var_6.z
    assert var_7 == 3



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/8 statements.
# Partially parsed test_immutable_set_multiple. Retrieved 8/9 statements.
# Partially parsed test_immutable_invalid_member_set. Retrieved 7/9 statements.
# Partially parsed test_immutable_frozen_members. Retrieved 9/12 statements.
# Partially parsed test_immutable_empty_members. Retrieved 5/6 statements.
# Partially parsed test_immutable_inheritance. Retrieved 7/13 statements.
# Partially parsed test_immutable_no_kwargs_returns_self. Retrieved 6/7 statements.


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
    var_6 = 10
    var_7 = 20

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)
    var_6 = 3
    var_7 = "'z' is not a member"

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
    var_9 = "Cannot set frozen members 'id_'"

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
    var_1 = 'Base'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = tuple()
    var_4 = 5
    var_5 = 10
    var_6 = 1

import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = var_2(var_3, var_4)



