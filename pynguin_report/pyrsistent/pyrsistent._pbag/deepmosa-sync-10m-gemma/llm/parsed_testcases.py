####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_remove_existing_element_multiple_count. Retrieved 7/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pbag as module_1

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = module_0.pmap(var_1)
    var_3 = module_1.PBag(var_2)
    var_4 = var_3.remove(var_0)
    var_5 = 1
    var_6 = bool(1 not in var_4)
    assert var_6 is True
    var_7 = len(var_4)
    assert var_7 == 0

import pyrsistent._pmap as module_0
import pyrsistent._pbag as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.PBag(var_3)
    var_5 = var_4.remove(var_0)
    var_6 = len(var_5)
    assert var_6 == 1

import pyrsistent._pmap as module_0
import pyrsistent._pbag as module_1

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = module_0.pmap(var_1)
    var_3 = module_1.PBag(var_2)
    var_4 = 2
    var_5 = var_3.remove(var_4)
    var_6 = 'KeyError not raised'
    var_7 = AssertionError(var_6)

import pyrsistent._pmap as module_0
import pyrsistent._pbag as module_1

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = module_0.pmap(var_1)
    var_3 = module_1.PBag(var_2)
    var_4 = var_3.remove(var_0)
    var_5 = 1
    var_6 = bool(1 in var_3._counts)
    assert var_6 is True
    var_7 = 1
    var_8 = bool(1 not in var_4._counts)
    assert var_8 is True



