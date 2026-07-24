####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = '__contains__'
    var_3 = '__getitem__'
    var_4 = 'a'
    var_5 = lambda s, k: k == var_4
    var_6 = 'val_a'
    var_7 = lambda s, k: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.PMapItems(var_12)
    var_14 = 'a'
    var_15 = 'val_a'
    var_16 = (var_14, var_15)
    var_17 = bool(('a', 'val_a') in var_13)
    assert var_17 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = '__contains__'
    var_3 = '__getitem__'
    var_4 = 'a'
    var_5 = lambda s, k: k == var_4
    var_6 = 'val_a'
    var_7 = lambda s, k: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.PMapItems(var_12)
    var_14 = 'a'
    var_15 = 'wrong_val'
    var_16 = (var_14, var_15)
    var_17 = bool(('a', 'wrong_val') not in var_13)
    assert var_17 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = '__contains__'
    var_3 = '__getitem__'
    var_4 = False
    var_5 = lambda s, k: var_4
    var_6 = None
    var_7 = lambda s, k: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.PMapItems(var_12)
    var_14 = 'b'
    var_15 = 'val_b'
    var_16 = (var_14, var_15)
    var_17 = bool(('b', 'val_b') not in var_13)
    assert var_17 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = '__contains__'
    var_3 = '__getitem__'
    var_4 = True
    var_5 = lambda s, k: var_4
    var_6 = lambda s, k: var_4
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = module_1.PMapItems(var_11)
    var_13 = 123
    var_14 = bool(123 not in var_12)
    assert var_14 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = '__contains__'
    var_3 = '__getitem__'
    var_4 = True
    var_5 = lambda s, k: var_4
    var_6 = lambda s, k: var_4
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = module_1.PMapItems(var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = bool(('a', 'b', 'c') not in var_12)
    assert var_17 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_pmap_values_eq_identity.
# Failed to parse test_pmap_values_eq_different_instance.
# Failed to parse test_pmap_values_eq_with_other_type.




# Parsed testcases at query #3
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = []
    var_8 = 'z'
    var_9 = 30
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_6, var_7, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._buckets
    var_18 = bool(var_16._buckets == var_12)
    assert var_18 is True



# Parsed testcases at query #4
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = (var_2, var_9)
    var_11 = [var_5, var_10]
    var_12 = 2
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 2
    var_17 = var_15._buckets
    var_18 = bool(var_15._buckets == var_11)
    assert var_18 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets[0]
    assert var_11 is None
    var_12 = var_9._buckets[1]
    var_13 = bool(var_9._buckets[1] == [('key', 'val')])
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_eq_mapping_proxy. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 != var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}



# Parsed testcases at query #6
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 2, 'c': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'b'
    var_7 = 'a'
    var_8 = {var_6: var_1, var_7: var_0}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != None)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._buckets[0][0][0]
    assert var_11 == 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = [var_6, var_5]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = 2
    var_10 = [var_9, var_5]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = bool(var_8 != var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 2
    var_7 = (var_1, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_8, var_0]
    var_10 = [var_2, var_5]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = [var_2, var_9]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = bool(var_12 != var_15)
    assert var_16 is True



# Parsed testcases at query #9
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._size
    assert var_9 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._buckets
    var_10 = bool(var_8._buckets == var_5)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = (var_2, var_9)
    var_11 = [var_5, var_10]
    var_12 = 2
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 2
    var_17 = var_15._buckets
    var_18 = bool(var_15._buckets == var_11)
    assert var_18 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = (var_2, var_9)
    var_11 = [var_5, var_10]
    var_12 = [var_7, var_11]
    var_13 = {}
    var_14 = module_0.PMap(*var_12, **var_13)
    var_15 = var_14._size
    assert var_15 == 2
    var_16 = var_14._buckets
    var_17 = bool(var_14._buckets == var_11)
    assert var_17 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == var_1)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == var_1)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = 2
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_turbo_mapping_with_unsizeable_initial. Retrieved 1/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 3
    var_11 = var_8._size
    assert var_11 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'x': 10, 'y': 20})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 16

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = var_6._buckets
    var_9 = len(var_8)
    assert var_9 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = var_2._buckets
    var_5 = len(var_4)
    assert var_5 == 8

def test_case_0():
    var_0 = None



# Parsed testcases at query #14
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3
    var_13 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['x']
    assert var_10 == 10
    var_11 = var_8['y']
    assert var_11 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 20
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1
    var_7 = var_4._buckets
    var_8 = len(var_7)
    assert var_8 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 == 1
    var_8 = var_5['b']
    assert var_8 == 2



# Parsed testcases at query #15
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = (var_2, var_9)
    var_11 = [var_5, var_10]
    var_12 = 2
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 2
    var_17 = var_15._buckets
    var_18 = bool(var_15._buckets == var_11)
    assert var_18 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == [None, None])
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]



# Parsed testcases at query #17
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 10
    var_3 = var_1 * var_2
    var_4 = 0
    var_5 = [var_4, var_3]
    var_6 = {}
    var_7 = module_0.PMap(*var_5, **var_6)
    var_8 = var_7._size
    assert var_8 == 0
    var_9 = var_7._buckets
    var_10 = bool(var_7._buckets == var_3)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_0, var_7, var_0]
    var_9 = 2
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    assert var_13 == 2
    var_14 = var_12._buckets
    var_15 = bool(var_12._buckets == var_8)
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_update_with_merge_rightmost. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_merge_leftmost. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_addition. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_no_overlap. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_maps. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'c'
    var_9 = {var_7: var_1, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 'd'
    var_13 = 17
    var_14 = 35
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r



# Parsed testcases at query #19
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 != var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [('a', 1), ('b', 2)])
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false_on_non_iterable_arg. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_contains_invalid_arg_type_returns_false. Retrieved 3/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 123
    var_2 = 'not_a_tuple'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_contains_evaluates_false_on_ununpackable_arg. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #23
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'key2'
    var_6 = 'val2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8]
    var_10 = 2
    var_11 = [var_10, var_9]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = var_13._size
    var_15 = bool(var_13._size == var_10)
    assert var_15 is True
    var_16 = var_13._buckets
    var_17 = bool(var_13._buckets == var_9)
    assert var_17 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._size
    assert var_9 == 1
    var_10 = var_8._buckets
    var_11 = bool(var_8._buckets == var_5)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == var_1)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_0.PMap(*var_9, **var_10)
    var_12 = var_11._size
    assert var_12 == 2
    var_13 = var_11._buckets
    var_14 = bool(var_11._buckets == var_8)
    assert var_14 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_update_with_merging_logic. Retrieved 6/9 statements.
# Partially parsed test_pmap_update_with_leftmost_logic. Retrieved 7/9 statements.
# Partially parsed test_pmap_update_with_dict_input. Retrieved 9/11 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)
    var_9 = 3
    var_10 = 'c'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 3
    var_9 = 'a'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)
    var_12 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 10
    var_9 = 20
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = module_0.m(**var_8)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_with_predicate_evaluates_to_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 5
    var_9 = 'b'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [('a', 1), ('b', 2)])
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_contains_valid_item. Retrieved 5/17 statements.
# Partially parsed test_contains_invalid_key_value_pair. Retrieved 3/14 statements.
# Failed to parse test_contains_non_iterable_arg.
# Failed to parse test_contains_tuple_with_wrong_structure.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_update_with_merge_leftmost_behavior. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_custom_function_addition. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_multiple_maps_and_custom_logic. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_no_overlapping_keys. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_empty_maps. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 2
    var_7 = 'a'
    var_8 = 'c'
    var_9 = {var_7: var_6, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 'd'
    var_13 = 3
    var_14 = 4
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = {}
    var_5 = module_0.m(**var_4)
    var_6 = {}
    var_7 = lambda l, r: r



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_contains_valid_tuple. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)



# Parsed testcases at query #5
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = (var_0, var_2)
    var_7 = 'a'
    var_8 = 1
    var_9 = (var_7, var_8)
    var_10 = bool(('a', 1) in var_5)
    assert var_10 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_update_with_predicate_is_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 10
    var_9 = 'b'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)



# Parsed testcases at query #7
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 != var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_update_with_predicate_false_on_new_key. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: l + r
    var_9 = 'b'
    var_10 = 'a'



# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1
    var_7 = var_4._buckets
    var_8 = len(var_7)
    assert var_8 == 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = iter(var_6)
    var_8 = None
    var_9 = module_0._turbo_mapping(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_9['x']
    assert var_11 == 10
    var_12 = var_9['y']
    assert var_12 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = var_2._buckets
    var_5 = len(var_4)
    assert var_5 == 8

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 == 1
    var_8 = var_5['b']
    assert var_8 == 2
    var_9 = 0
    var_10 = var_5._buckets[var_9]
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #11
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 1) in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 3) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('c', 1) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = 'a'
    var_5 = bool('a' not in var_3)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = 1
    var_5 = bool(1 not in var_3)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = 'a'
    var_5 = bool('a' not in var_3)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = ()
    var_5 = bool(() not in var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]



# Parsed testcases at query #13
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != [('a', 1), ('b', 2)])
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]



# Parsed testcases at query #16
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._size
    assert var_9 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._buckets
    var_10 = bool(var_8._buckets == var_5)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9['key']
    assert var_10 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_5._size
    assert var_7 == 0



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_contains_returns_false_on_unpacking_error.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_turbo_mapping_exception_handling. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_contains_raises_exception_returns_false. Retrieved 2/12 statements.


def test_case_0():
    var_0 = None
    var_1 = 123



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #21
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = 'z'
    var_9 = 30
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_6, var_7, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._buckets
    var_18 = bool(var_16._buckets == var_12)
    assert var_18 is True



# Parsed testcases at query #22
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_0, var_8]
    var_10 = 2
    var_11 = [var_10, var_9]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = var_13._size
    assert var_14 == 2
    var_15 = var_13._buckets
    var_16 = bool(var_13._buckets == var_9)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = 1
    var_6 = [var_5, var_4]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._size
    assert var_9 == 1
    var_10 = var_8._buckets[0][0]
    var_11 = bool(var_8._buckets[0][0] == ('key', 'val'))
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0



# Parsed testcases at query #23
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'val1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'val2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 'key3'
    var_9 = 'val3'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_0, var_7, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    assert var_17 == 3
    var_18 = var_16._buckets
    var_19 = bool(var_16._buckets == var_12)
    assert var_19 is True



# Parsed testcases at query #24
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8]
    var_10 = 2
    var_11 = [var_10, var_9]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = var_13._size
    assert var_14 == 2
    var_15 = var_13._buckets
    var_16 = bool(var_13._buckets == var_9)
    assert var_16 is True
    var_17 = len(var_13)
    assert var_17 == 2



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_contains_valid_pair. Retrieved 5/16 statements.
# Partially parsed test_contains_invalid_value. Retrieved 3/14 statements.
# Partially parsed test_contains_invalid_format. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 2
    var_10 = (var_8, var_9)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 1
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_4,)
    var_6 = None
    var_7 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_predicate_false_when_key_not_in_evolver. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'b'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 3
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #28
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]
    var_7 = var_3 == var_6



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_with_does_not_trigger_key_error_on_new_key. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #31
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8._size
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3
    var_13 = var_8._buckets
    var_14 = len(var_13)
    assert var_14 == 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4['x']
    assert var_6 == 100
    var_7 = var_4._buckets
    var_8 = len(var_7)
    assert var_8 == 8

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 4
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = var_2._size
    assert var_3 == 0
    var_4 = var_2._buckets
    var_5 = len(var_4)
    assert var_5 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8._size
    assert var_9 == 2
    var_10 = var_8['key1']
    assert var_10 == 'val1'
    var_11 = var_8['key2']
    assert var_11 == 'val2'
    var_12 = var_8._buckets
    var_13 = len(var_12)
    assert var_13 == 5



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_contains_evaluates_false_on_uniterable_arg. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #33
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'val1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'val2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 'key3'
    var_9 = 'val3'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_0, var_7, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    assert var_17 == 3
    var_18 = var_16._buckets
    var_19 = bool(var_16._buckets == var_12)
    assert var_19 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pmap_items_contains_valid_tuple. Retrieved 5/21 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 5/19 statements.
# Partially parsed test_pmap_items_contains_non_iterable_arg. Retrieved 3/15 statements.
# Partially parsed test_pmap_items_contains_key_not_in_map. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 2
    var_7 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = 1
    var_5 = (var_3, var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_update_with_does_not_evaluate_true_for_new_keys. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #37
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True
    var_5 = bool(var_3 != None)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 2
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_6)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = [var_1, var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = var_8._size
    assert var_9 == 1
    var_10 = var_8._buckets
    var_11 = bool(var_8._buckets == var_5)
    assert var_11 is True



# Parsed testcases at query #39
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 1) in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 3) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('c', 1) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = bool('a' not in var_5)
    assert var_7 is True
    var_8 = 1
    var_9 = bool(1 not in var_5)
    assert var_9 is True
    var_10 = None
    var_11 = bool(None not in var_5)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)
    var_10 = bool(('a', 1, 'extra') not in var_5)
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



