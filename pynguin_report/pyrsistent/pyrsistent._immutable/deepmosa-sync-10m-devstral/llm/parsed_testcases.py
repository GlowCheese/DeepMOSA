####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/9 statements.
# Partially parsed test_immutable_inheritance. Retrieved 5/14 statements.
# Partially parsed test_immutable_frozen_members. Retrieved 9/12 statements.
# Partially parsed test_immutable_invalid_member. Retrieved 7/9 statements.
# Partially parsed test_immutable_empty_members. Retrieved 5/6 statements.


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
    assert var_5 == 'Immutable(x=1, y=2)'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_immutable_set_method. Retrieved 7/9 statements.
# Partially parsed test_immutable_inheritance. Retrieved 5/14 statements.
# Partially parsed test_immutable_frozen_members. Retrieved 9/12 statements.
# Partially parsed test_immutable_invalid_member. Retrieved 7/9 statements.


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
    var_7 = bool(False)
    assert var_7 is True



