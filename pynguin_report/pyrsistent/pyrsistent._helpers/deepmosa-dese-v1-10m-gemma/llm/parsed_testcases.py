####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 8/14 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 3/9 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_handles_complex_nesting. Retrieved 12/24 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {var_2: var_3}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 'b'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 18/26 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/8 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = [var_5]
    var_7 = (var_4, var_6)
    var_8 = {var_3: var_7}
    var_9 = [var_2, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = {var_10, var_11}
    var_13 = {var_0: var_9, var_1: var_12}
    var_14 = [var_5]
    var_15 = {var_10, var_11}
    var_16 = module_0.pset(var_15)
    var_17 = module_1.freeze(var_13)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.freeze(var_0)
    assert var_1 == 10
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 16/21 statements.
# Partially parsed test_freeze_strict_false_on_dict_values. Retrieved 10/13 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 'b'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_4, var_9]
    var_11 = [var_1, var_2]
    var_12 = {var_6: var_7}
    var_13 = module_0.pmap(var_12)
    var_14 = (var_5, var_13)
    var_15 = module_1.freeze(var_10)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.freeze(var_0)
    assert var_1 == 5
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = module_0.freeze(var_6, var_7)
    var_9 = [var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_list. Retrieved 16/24 statements.
# Partially parsed test_freeze_tuple. Retrieved 11/13 statements.
# Partially parsed test_freeze_dict. Retrieved 21/24 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 15/21 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = [var_0]
    var_11 = [var_1]
    var_12 = [var_10, var_11]
    var_13 = module_0.freeze(var_12)
    var_14 = [var_0]
    var_15 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = [var_1]
    var_5 = (var_0, var_4)
    var_6 = module_0.freeze(var_5)
    var_7 = [var_1]
    var_8 = (var_0,)
    var_9 = (var_8,)
    var_10 = module_0.freeze(var_9)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = 'tuple'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_1.pmap(var_7)
    var_9 = [var_2, var_3]
    var_10 = {var_0: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = [var_2, var_3]
    var_13 = 3
    var_14 = {var_1: var_13}
    var_15 = {var_0: var_14}
    var_16 = module_0.freeze(var_15)
    var_17 = {var_1: var_13}
    var_18 = module_1.pmap(var_17)
    var_19 = {var_0: var_18}
    var_20 = module_1.pmap(var_19)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = (var_0, var_1)
    var_8 = {var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = (var_0, var_1)
    var_11 = {var_10}
    var_12 = module_1.pset(var_11)

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = {var_1: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = [var_0, var_6, var_9]
    var_11 = [var_3]
    var_12 = {var_7, var_8}
    var_13 = module_0.pset(var_12)
    var_14 = module_1.freeze(var_10)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = False
    var_6 = module_1.freeze(var_4, var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 11/18 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/14 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 8/22 statements.
# Partially parsed test_mutant_preserves_simple_types. Retrieved 3/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'b'
    var_7 = {var_6: var_1}
    var_8 = [var_0, var_1, var_2]
    var_9 = {var_6: var_1}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = 0
    var_7 = [var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_strict_pmap_is_true. Retrieved 8/13 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_4}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_list. Retrieved 13/18 statements.
# Partially parsed test_freeze_dict. Retrieved 17/20 statements.
# Partially parsed test_freeze_tuple. Retrieved 11/13 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 23/28 statements.
# Partially parsed test_freeze_strict_false_on_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_pvector_input. Retrieved 4/7 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = []
    var_11 = module_0.freeze(var_10)
    var_12 = []

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_2, var_3]
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_8, var_1: var_11}
    var_13 = module_0.freeze(var_12)
    var_14 = [var_2, var_3]
    var_15 = {var_9: var_10}
    var_16 = module_1.pmap(var_15)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = 3
    var_5 = [var_1, var_4]
    var_6 = (var_0, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_1, var_4]
    var_9 = ()
    var_10 = module_0.freeze(var_9)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = set()
    var_8 = module_0.freeze(var_7)
    var_9 = module_1.pset()

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 4
    var_8 = 'c'
    var_9 = 5
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = 6
    var_13 = 7
    var_14 = {var_12, var_13}
    var_15 = [var_6, var_11, var_14]
    var_16 = [var_3, var_4]
    var_17 = {var_8: var_9}
    var_18 = module_0.pmap(var_17)
    var_19 = (var_7, var_18)
    var_20 = {var_12, var_13}
    var_21 = module_1.pset(var_20)
    var_22 = module_2.freeze(var_15)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_1]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_is_decorator. Retrieved 18/35 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = 2
    var_9 = [var_1, var_8]
    var_10 = 'key'
    var_11 = 3
    var_12 = [var_11]
    var_13 = module_1.pset(var_12)
    var_14 = {var_10: var_13}
    var_15 = [var_1, var_8]
    var_16 = [var_11]
    var_17 = module_1.pset(var_16)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_defaultdict_type. Retrieved 7/11 statements.
# Partially parsed test_freeze_with_dict_input. Retrieved 4/6 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = (var_1,)
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_handles_kwargs_recursion. Retrieved 8/20 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'b'
    var_7 = {var_6: var_1}

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = 'data'
    var_7 = 0

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 2
    var_3 = (var_0, var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_list. Retrieved 14/21 statements.
# Partially parsed test_freeze_dict. Retrieved 22/25 statements.
# Partially parsed test_freeze_tuple. Retrieved 15/19 statements.
# Partially parsed test_freeze_nested_complex. Retrieved 23/28 statements.
# Partially parsed test_freeze_strict_parameter. Retrieved 8/13 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None
    var_6 = 3.14
    var_7 = module_0.freeze(var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = [var_0]
    var_11 = [var_10]
    var_12 = module_0.freeze(var_11)
    var_13 = [var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = 'tuple'
    var_7 = module_1.pmap()
    var_8 = {var_0: var_2, var_6: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = [var_2, var_3]
    var_11 = {var_0: var_10}
    var_12 = module_0.freeze(var_11)
    var_13 = [var_2, var_3]
    var_14 = 3
    var_15 = {var_1: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.freeze(var_16)
    var_18 = {var_1: var_14}
    var_19 = module_1.pmap(var_18)
    var_20 = {var_0: var_19}
    var_21 = module_1.pmap(var_20)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = 3
    var_5 = [var_1, var_4]
    var_6 = (var_0, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_1, var_4]
    var_9 = (var_0,)
    var_10 = [var_1]
    var_11 = (var_9, var_10)
    var_12 = module_0.freeze(var_11)
    var_13 = (var_0,)
    var_14 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = (var_0, var_1)
    var_8 = {var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = (var_0, var_1)
    var_11 = {var_10}
    var_12 = module_1.pset(var_11)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 4
    var_8 = 'c'
    var_9 = 5
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = 6
    var_13 = 7
    var_14 = {var_12, var_13}
    var_15 = [var_6, var_11, var_14]
    var_16 = [var_3, var_4]
    var_17 = {var_8: var_9}
    var_18 = module_0.pmap(var_17)
    var_19 = (var_7, var_18)
    var_20 = {var_12, var_13}
    var_21 = module_1.pset(var_20)
    var_22 = module_2.freeze(var_15)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_5]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_dict_is_true. Retrieved 4/6 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/18 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/7 statements.
# Partially parsed test_mutant_handles_tuple_recursively. Retrieved 4/11 statements.
# Partially parsed test_mutant_handles_kwargs_as_dict. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 0

def test_case_0():
    var_0 = 'val1'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'key2'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 2/11 statements.
# Partially parsed test_mutant_isolates_mutation_in_function. Retrieved 6/12 statements.
# Partially parsed test_mutant_handles_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/28 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}

def test_case_0():
    var_0 = 0
    var_1 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'c'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_0]
    var_6 = {var_2: var_3}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = [var_2, var_6]
    var_8 = 4
    var_9 = 5
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 11/26 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 2
    var_7 = [var_1, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = module_1.pset()
    var_10 = [var_1, var_6]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 10/17 statements.
# Partially parsed test_mutant_ensures_output_is_frozen. Retrieved 1/6 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 3/10 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = [var_0]
    var_6 = {var_4: var_5}
    var_7 = 'b'
    var_8 = [var_1]
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap()

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = {var_0: var_6, var_1: var_9}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_input_args. Retrieved 4/9 statements.
# Partially parsed test_mutant_does_not_change_value_of_immutable_types. Retrieved 1/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 5



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 14/24 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 5/12 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = 2
    var_7 = [var_1, var_6]
    var_8 = [var_1, var_6]
    var_9 = [var_1, var_6]
    var_10 = module_1.pset(var_9)
    var_11 = [var_1, var_6]
    var_12 = module_1.pset(var_11)
    var_13 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 7/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 3/9 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 16/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = (var_9, var_10)
    var_12 = 6
    var_13 = 7
    var_14 = {var_12, var_13}
    var_15 = {var_0: var_8, var_1: var_11, var_2: var_14}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 4/9 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'original'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 4/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 12/19 statements.
# Partially parsed test_mutant_preserves_return_value_structure. Retrieved 5/14 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 1/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'y'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_3}
    var_10 = [var_0, var_1, var_9]
    var_11 = {var_6: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = [var_2, var_3]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_handles_kwargs. Retrieved 4/16 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap()
    var_5 = var_4.__class__
    var_6 = module_1.pset()
    var_7 = var_6.__class__

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = var_3.__class__

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'key'
    var_2 = module_0.pmap()
    var_3 = var_2.__class__



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_preserves_functionality_and_freezes_inputs. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_0}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 9/17 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_4]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = 2
    var_4 = [var_0, var_3]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 7/15 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/16 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/7 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_preserves_primitive_types. Retrieved 3/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 1
    var_1 = 'inner'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 6/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_logic. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 11/22 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_handles_kwargs_freezing. Retrieved 8/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = {var_1: var_2}
    var_6 = {var_1: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = 2
    var_9 = 10
    var_10 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'val'
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_3: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 7/14 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 8/15 statements.
# Partially parsed test_mutant_deep_freezing_of_return_value. Retrieved 5/16 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]
    var_6 = {var_0: var_5}
    var_7 = [var_1, var_2]

def test_case_0():
    var_0 = 'key'
    var_1 = 'other'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'inner'
    var_2 = 2
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = [var_0, var_4]
    var_6 = [var_2]



# Parsed testcases at query #36
#--------------------------




import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = module_0.pmap()
    var_7 = var_6.__class__
    var_8 = isinstance(var_5, var_7)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_decorator_does_not_mutate_arguments. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 13/22 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)
    var_7 = [var_1, var_4]
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_1, var_4]
    var_11 = module_1.pset(var_10)
    var_12 = [var_1, var_4]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_preserves_functionality. Retrieved 14/21 statements.
# Partially parsed test_mutant_freezes_inputs. Retrieved 5/9 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = 2
    var_7 = [var_1, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = 3
    var_10 = [var_9]
    var_11 = module_1.pset(var_10)
    var_12 = [var_1]
    var_13 = module_1.pset(var_12)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 12/19 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 3/9 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 4/11 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = {var_1: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = 2
    var_8 = [var_2, var_7]
    var_9 = module_1.pset(var_8)
    var_10 = [var_2, var_7]
    var_11 = module_1.pset(var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'data'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_thaw_pset. Retrieved 3/6 statements.
# Partially parsed test_thaw_pvector_nested. Retrieved 5/9 statements.
# Partially parsed test_thaw_pmap_nested. Retrieved 4/8 statements.
# Partially parsed test_thaw_tuple_recursive. Retrieved 5/9 statements.
# Partially parsed test_thaw_strict_false_list. Retrieved 3/9 statements.
# Partially parsed test_thaw_strict_false_dict. Retrieved 3/9 statements.
# Partially parsed test_thaw_mixed_containers. Retrieved 10/15 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = (var_3,)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = False

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_1,)
    var_3 = 3
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_0}
    var_7 = (var_1,)
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = [var_8, var_3]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_containers. Retrieved 10/11 statements.
# Partially parsed test_freeze_list_recursive. Retrieved 15/22 statements.
# Partially parsed test_freeze_dict_recursive. Retrieved 14/22 statements.
# Partially parsed test_freeze_tuple_recursive. Retrieved 14/18 statements.
# Partially parsed test_freeze_strict_false_behavior. Retrieved 5/7 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_0.freeze(var_2)
    var_4 = module_1.pmap()
    var_5 = set()
    var_6 = module_0.freeze(var_5)
    var_7 = module_2.pset()
    var_8 = ()
    var_9 = module_0.freeze(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = (var_1, var_2)
    var_10 = [var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = {var_7: var_0}
    var_13 = module_1.pmap(var_12)
    var_14 = []

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_3, var_4]
    var_9 = 'inner'
    var_10 = []
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.freeze(var_12)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]
    var_6 = [var_0]
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = (var_6, var_8)
    var_10 = module_0.freeze(var_9)
    var_11 = [var_0]
    var_12 = {var_7: var_1}
    var_13 = module_1.pmap(var_12)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = list(var_1)
    var_3 = [var_2]
    var_4 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_list. Retrieved 19/25 statements.
# Partially parsed test_freeze_tuple. Retrieved 17/20 statements.
# Partially parsed test_freeze_dict. Retrieved 17/20 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 22/27 statements.
# Partially parsed test_freeze_empty_containers. Retrieved 12/13 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_0]
    var_7 = [var_1]
    var_8 = [var_6, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_0]
    var_11 = [var_1]
    var_12 = 'a'
    var_13 = {var_12: var_0}
    var_14 = [var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = {var_12: var_0}
    var_17 = module_1.pmap(var_16)
    var_18 = [var_17]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = [var_0]
    var_5 = [var_1]
    var_6 = (var_4, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0]
    var_9 = [var_1]
    var_10 = 'a'
    var_11 = {var_10: var_0}
    var_12 = (var_11,)
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0}
    var_15 = module_1.pmap(var_14)
    var_16 = (var_15,)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_2, var_3]
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_8, var_1: var_11}
    var_13 = module_0.freeze(var_12)
    var_14 = [var_2, var_3]
    var_15 = {var_9: var_10}
    var_16 = module_1.pmap(var_15)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = (var_0, var_1)
    var_8 = 4
    var_9 = (var_2, var_8)
    var_10 = {var_7, var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = (var_0, var_1)
    var_13 = (var_2, var_8)
    var_14 = {var_12, var_13}
    var_15 = module_1.pset(var_14)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = (var_8, var_9)
    var_11 = 6
    var_12 = 7
    var_13 = {var_11, var_12}
    var_14 = [var_7, var_10, var_13]
    var_15 = {var_3: var_4}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_1, var_2, var_16]
    var_18 = (var_8, var_9)
    var_19 = {var_11, var_12}
    var_20 = module_1.pset(var_19)
    var_21 = module_2.freeze(var_14)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_0.freeze(var_3)
    var_5 = {}
    var_6 = module_1.pmap(var_5)
    var_7 = ()
    var_8 = module_0.freeze(var_7)
    var_9 = set()
    var_10 = module_0.freeze(var_9)
    var_11 = module_2.pset()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 9/17 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 1/7 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 5/11 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 6/16 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = [var_1]
    var_5 = module_0.freeze(var_4)
    var_6 = 3
    var_7 = [var_6]
    var_8 = len(var_5)
    assert var_8 == 1

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_deep_freeze_nested_structures. Retrieved 9/20 statements.
# Partially parsed test_mutant_handles_tuple_recursion. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 6/8 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 20/23 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = [var_2, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_8, var_1: var_11}
    var_13 = {var_4: var_5}
    var_14 = module_0.pmap(var_13)
    var_15 = (var_3, var_14)
    var_16 = [var_2, var_15]
    var_17 = [var_9, var_10]
    var_18 = module_1.pset(var_17)
    var_19 = module_2.freeze(var_12)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 10/14 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 7/15 statements.
# Partially parsed test_mutant_deep_freeze. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_1, var_5]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = [var_3]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = {var_0: var_6, var_1: var_10}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_list. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_set. Retrieved 4/6 statements.
# Partially parsed test_freeze_recursive_list. Retrieved 5/9 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.freeze(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = module_0.freeze(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.freeze(var_0)
    assert var_1 == 5

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0.freeze(var_2)
    var_4 = [var_0]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 14/25 statements.
# Partially parsed test_mutant_preserves_primitives. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'inner_key'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = 0
    var_12 = {var_6, var_7}
    var_13 = module_0.pset(var_12)

def test_case_0():
    var_0 = 10
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_list_to_pvector. Retrieved 8/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/16 statements.
# Partially parsed test_freeze_dict_to_pmap. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple_to_tuple_recursive. Retrieved 12/16 statements.
# Partially parsed test_freeze_set_to_pset. Retrieved 9/10 statements.
# Partially parsed test_freeze_list_of_dicts. Retrieved 13/14 statements.
# Partially parsed test_freeze_strict_false_behavior_on_dict. Retrieved 7/10 statements.
# Partially parsed test_freeze_empty_containers. Retrieved 13/14 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.freeze(var_5)
    var_7 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = [var_1, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0]
    var_9 = [var_3]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.freeze(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]
    var_6 = (var_0,)
    var_7 = [var_1]
    var_8 = (var_6, var_7)
    var_9 = module_0.freeze(var_8)
    var_10 = (var_0,)
    var_11 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_0.freeze(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_1.pset(var_7)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_1.pmap(var_8)
    var_10 = {var_3: var_4}
    var_11 = module_1.pmap(var_10)
    var_12 = [var_9, var_11]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_0.freeze(var_3)
    var_5 = {}
    var_6 = module_1.pmap(var_5)
    var_7 = ()
    var_8 = module_0.freeze(var_7)
    var_9 = set()
    var_10 = module_0.freeze(var_9)
    var_11 = []
    var_12 = module_2.pset(var_11)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_decorator_returns_function. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/16 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 1/8 statements.
# Partially parsed test_mutant_preserves_logic_with_mutation_inside. Retrieved 2/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/18 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 19/27 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'inner'
    var_2 = 2
    var_3 = [var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = (var_9, var_10)
    var_12 = 6
    var_13 = 7
    var_14 = {var_12, var_13}
    var_15 = {var_0: var_8, var_1: var_11, var_2: var_14}
    var_16 = {var_5: var_6}
    var_17 = {var_12, var_13}
    var_18 = module_0.pset(var_17)

def test_case_0():
    var_0 = 10



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 12/19 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 4/9 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = 2
    var_7 = [var_1, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = [var_1, var_6]
    var_10 = module_1.pset(var_9)
    var_11 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_predicate_evaluates_to_false. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 11/22 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = 10
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_decorator_works. Retrieved 8/19 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 12/18 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 4/10 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = 2
    var_7 = [var_1, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = [var_1, var_6]
    var_10 = module_1.pset(var_9)
    var_11 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/13 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_preserves_logic_with_frozen_inputs. Retrieved 7/14 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 4/12 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_4: var_0}

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 9/20 statements.
# Partially parsed test_mutant_preserves_primitives. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = (var_1, var_6)
    var_8 = [var_0, var_7]

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_mutant_decorator_is_callable.




# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_4}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/18 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Failed to parse test_mutant_handles_empty_inputs.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_mutant_decorator_returns_function.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/14 statements.
# Partially parsed test_mutant_preserves_logic. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/19 statements.
# Partially parsed test_mutant_handles_no_args. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_decorates_function. Retrieved 21/37 statements.
# Partially parsed test_mutant_returns_frozen_value. Retrieved 5/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 'nested'
    var_7 = 'list'
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = 'a'
    var_11 = {var_10: var_3}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_3]
    var_14 = {var_6: var_7}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_1}
    var_17 = module_0.pmap(var_16)
    var_18 = [var_3, var_4]
    var_19 = {var_6: var_7}
    var_20 = module_0.pmap(var_19)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 4/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 18/35 statements.
# Partially parsed test_mutant_preserves_primitives. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = 'inner'
    var_6 = 3
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = (var_10, var_11)
    var_13 = 6
    var_14 = 7
    var_15 = {var_13, var_14}
    var_16 = {var_0: var_9, var_1: var_12, var_2: var_15}
    var_17 = 0

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 8/17 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 1/6 statements.
# Partially parsed test_mutant_handles_tuple_recursion. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 10
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = [var_4]
    var_6 = {var_0: var_5}
    var_7 = 0

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_decorator_returns_frozen_value. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 11/20 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = module_0.pmap()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_returns. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_mutable_inputs. Retrieved 5/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



