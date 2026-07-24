####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_with_list_containing_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_with_tuple_containing_list. Retrieved 5/7 statements.
# Partially parsed test_freeze_with_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_with_non_strict_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_with_defaultdict. Retrieved 5/8 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = []

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

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
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, (2, 3)))
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = [var_4, var_1, var_2]

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_8 = [var_1, var_2]
    var_9 = {var_0: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = {var_1: var_2}
    var_7 = [var_0, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Failed to parse test_mutant_with_simple_return.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 5/8 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_pset_input. Retrieved 6/9 statements.
# Failed to parse test_mutant_preserves_none.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 5/10 statements.


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
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = {var_0, var_1}
    var_4 = module_0.pset(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = [var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'c'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_1]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_2, var_12]
    var_14 = {var_7, var_8}
    var_15 = module_1.pset(var_14)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]

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
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = module_0.pset(var_2)
    var_4 = {var_0, var_1}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 42
    var_1 = 'string'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 7/12 statements.
# Partially parsed test_mutant_nested. Retrieved 9/16 statements.
# Partially parsed test_mutant_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_no_mutation. Retrieved 9/19 statements.
# Partially parsed test_mutant_strict_false. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]

def test_case_0():
    var_0 = 'count'
    var_1 = 'items'
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = [var_3, var_4, var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = 'z'
    var_8 = {var_0: var_1, var_3: var_4, var_7: var_6}
    var_9 = module_0.pmap(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = 3
    var_7 = [var_1, var_2, var_6]
    var_8 = bool(var_4 == {'a': [1, 2]})
    assert var_8 is True
    var_9 = [var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = [var_0, var_1, var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_with_defaultdict. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 16/30 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/17 statements.
# Partially parsed test_mutant_with_pvector_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_pmap_argument. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_pset_argument. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_0, var_1, var_2}
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1, var_2, var_7}
    var_14 = module_1.pset(var_13)
    var_15 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 42
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = {var_0, var_1, var_2, var_5}
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_predicate_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = [var_0]
    var_2 = True
    var_3 = {}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_non_container_types. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = [var_4, var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 6
    var_5 = 9
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_converts_defaultdict_to_pmap. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 12/13 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = module_1.pset(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = (var_2, var_3)
    var_9 = {var_1: var_8}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_0, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 11/17 statements.
# Partially parsed test_mutant_nested. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_list_operations. Retrieved 6/11 statements.
# Partially parsed test_mutant_set_operations. Retrieved 7/11 statements.
# Partially parsed test_mutant_tuple_operations. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = {var_5: var_0, var_7: var_1}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 6
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_4 == {'nested': {'value': 5}})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = 'z'
    var_8 = {var_0: var_1, var_3: var_4, var_7: var_6}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)
    var_7 = bool(var_3 == {1, 2, 3})
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = bool(var_3 == (1, 2, 3))
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_decorator_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 15/22 statements.
# Partially parsed test_mutant_decorator_with_mixed_args_and_kwargs. Retrieved 15/21 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 9/12 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 7/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = {var_5: var_0, var_7: var_1}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = 'result'
    var_6 = [var_1, var_2]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_8: var_1}
    var_10 = {var_7: var_9}
    var_11 = {var_8: var_1}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_7: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = {var_5: var_14}
    var_16 = module_0.pmap(var_15)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]
    var_7 = 'y'
    var_8 = 4
    var_9 = {var_7: var_8}
    var_10 = 'x'
    var_11 = {var_7: var_8}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_10: var_12}
    var_14 = module_0.pmap(var_13)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = 3
    var_4 = 4
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'kwargs'
    var_9 = [var_1]
    var_10 = 'c'
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_10: var_12}
    var_14 = module_0.pmap(var_13)

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 'set'
    var_5 = {var_0, var_1, var_2}
    var_6 = module_0.pset(var_5)
    var_7 = {var_4: var_6}
    var_8 = module_1.pmap(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = 'tuple'
    var_6 = [var_1, var_2]

def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_defaultdict_strict_mode. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = {var_0: var_5, var_1: var_3}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 11/17 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 12/13 statements.
# Partially parsed test_freeze_pvector_with_strict_true. Retrieved 5/7 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = {var_1: var_6}
    var_8 = [var_0, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_3, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = False
    var_8 = module_0.freeze(var_6, var_7)
    var_9 = [var_2, var_3]
    var_10 = {var_1: var_9}
    var_11 = [var_0, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_primitives. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_list_with_nested_lists. Retrieved 7/10 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_already_persistent. Retrieved 15/18 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 15/16 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_1.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = (var_0, var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = [var_0, var_1]
    var_5 = module_0.pset(var_4)
    var_6 = module_1.freeze(var_5)
    var_7 = [var_0, var_1]
    var_8 = module_0.pset(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True
    var_10 = 'a'
    var_11 = {var_10: var_0}
    var_12 = module_2.pmap(var_11)
    var_13 = module_1.freeze(var_12)
    var_14 = {var_10: var_0}
    var_15 = module_2.pmap(var_14)
    var_16 = bool(var_13 == var_15)
    assert var_16 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_1, var_2]
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = bool(var_6 == var_9)
    assert var_10 is True
    var_11 = {var_0: var_2}
    var_12 = [var_1, var_11]
    var_13 = module_0.freeze(var_12, var_5)
    var_14 = {var_0: var_2}
    var_15 = [var_1, var_14]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_dict_conversion. Retrieved 17/21 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = 'd'
    var_8 = 4
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_3, var_1: var_6, var_2: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = var_11['a']
    assert var_12 == 1
    var_13 = var_11[var_1]
    var_14 = [var_4, var_5]
    var_15 = var_11['b']
    var_16 = var_11[var_2]
    var_17 = {var_7: var_8}
    var_18 = module_1.pmap(var_17)
    var_19 = var_11['c']
    var_20 = bool(var_11['c'] == var_18)
    assert var_20 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_mutant_decorator_returns_callable.


def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #24
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.freeze(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 13/19 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'd'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_1, var_2]
    var_11 = {var_4: var_5}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'x'
    var_5 = 'y'
    var_6 = [var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_2, var_12]
    var_14 = {var_7, var_8}
    var_15 = module_1.pset(var_14)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 13/19 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 11/16 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 14/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = 'b'
    var_10 = {var_9: var_1}
    var_11 = {var_7: var_0, var_9: var_1}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 'nested'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = {var_3: var_4}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_1, var_2, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)
    var_8 = [var_1, var_2]
    var_9 = {var_4: var_5}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = []
    var_3 = module_1.pset()
    var_4 = ()

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_0, var_1]
    var_10 = {var_3: var_4}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_6, var_7}
    var_13 = module_1.pset(var_12)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_with_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_with_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_non_strict_mode. Retrieved 13/16 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_1.freeze(var_5)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.pmap(var_8)
    var_10 = module_1.freeze(var_9, var_4)
    var_11 = {var_6: var_0, var_7: var_1}
    var_12 = module_0.pmap(var_11)
    var_13 = bool(var_10 == var_12)
    assert var_13 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = bool(not (not False and True))
    assert var_0 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_defaultdict. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_converts_dict_to_pmap. Retrieved 8/9 statements.


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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_pvector_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 14/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'kwargs'
    var_7 = 'c'
    var_8 = 'd'
    var_9 = {var_7: var_2, var_8: var_3}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_4: var_0, var_5: var_1, var_6: var_10}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_1, var_2, var_4, var_5]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'value'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = 5
    var_10 = {var_0: var_5, var_1: var_8, var_2: var_9}
    var_11 = [var_3, var_4, var_9]
    var_12 = [var_6, var_7, var_9]
    var_13 = module_0.pset(var_12)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = {var_5: var_0, var_7: var_1}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'nested'
    var_7 = [var_1, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'y'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'kwargs'
    var_10 = [var_1, var_2]
    var_11 = 'x'
    var_12 = {var_4: var_5}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_11: var_13}
    var_15 = module_0.pmap(var_14)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 42
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 'hello'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_with_strict_true_converts_pvector_elements. Retrieved 9/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = {var_1: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_5, var_7]



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_freeze_set_returns_pset. Retrieved 5/6 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)



# Parsed testcases at query #16
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = var_1(var_2)
    assert var_3 == 1



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_6, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_1, var_2]



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_func. Retrieved 1/3 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = 5

def test_case_0():
    var_0 = 1
    var_1 = 5



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_positional_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 5/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/10 statements.
# Partially parsed test_mutant_returns_frozen_structures. Retrieved 4/9 statements.
# Partially parsed test_mutant_returns_frozen_dict. Retrieved 6/10 statements.
# Partially parsed test_mutant_returns_frozen_set. Retrieved 5/9 statements.
# Partially parsed test_mutant_returns_frozen_tuple. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 3
    var_1 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_primitives. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_list_with_nested_lists. Retrieved 7/10 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 16/19 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = (var_0, var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = [var_4, var_1, var_2]

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = 'a'
    var_9 = {var_0, var_1}
    var_10 = {var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_1.freeze(var_11, var_5)
    var_13 = {var_0, var_1}
    var_14 = {var_8: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 11/17 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_1.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = {var_1: var_6}
    var_8 = [var_0, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_3, var_4]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 11/17 statements.
# Partially parsed test_freeze_pvector_without_strict. Retrieved 5/7 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = {var_1: var_6}
    var_8 = [var_0, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #28
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_mutant_with_empty_args_and_kwargs.
# Partially parsed test_mutant_with_simple_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_simple_kwargs. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 10
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = {var_0, var_1, var_2, var_4, var_5}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_5]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = module_1.thaw(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.s(*var_3)
    var_5 = module_1.thaw(var_4)
    var_6 = bool(var_5 == {1, 2, 3})
    assert var_6 is True

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = (var_0, var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = bool(var_6 == (1, [2, 3]))
    assert var_7 is True

import pyrsistent._pvector as module_0
import pyrsistent._pmap as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = module_0.v(*var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_1, var_7: var_5}
    var_9 = module_1.m(**var_8)
    var_10 = [var_0, var_9]
    var_11 = module_0.v(*var_10)
    var_12 = module_2.thaw(var_11)
    var_13 = bool(var_12 == [1, {'a': 2, 'b': [3, 4]}])
    assert var_13 is True

import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]
    var_6 = module_1.v(*var_5)
    var_7 = False
    var_8 = module_2.thaw(var_6, var_7)
    var_9 = 'a'
    var_10 = {var_9: var_1}
    var_11 = module_0.m(**var_10)
    var_12 = [var_0, var_11]
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.thaw(var_3, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.thaw(var_4, var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_2
import pyrsistent._pset as module_3

def test_case_0():
    var_0 = []
    var_1 = module_0.v(*var_0)
    var_2 = module_1.thaw(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = {}
    var_5 = module_2.m(**var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = []
    var_9 = module_3.s(*var_8)
    var_10 = module_1.thaw(var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'hello'
    var_4 = None
    var_5 = module_0.thaw(var_4)
    assert var_5 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_nested_structures. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_pvector_with_strict_true. Retrieved 9/12 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.freeze(var_3)
    var_5 = module_0.freeze(var_1)
    var_6 = [var_4, var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = module_1.freeze(var_6)
    var_9 = module_1.freeze(var_3)
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.freeze(var_4)
    var_6 = module_0.freeze(var_1)
    var_7 = module_0.freeze(var_2)
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_with_strict_true_converts_dict_to_pmap. Retrieved 9/10 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = {var_0: var_5, var_1: var_3}
    var_8 = module_1.pmap(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_converts_dict_to_pmap. Retrieved 8/9 statements.


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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_with_defaultdict. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_no_return. Retrieved 1/4 statements.
# Partially parsed test_mutant_with_set_arguments. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_arguments. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_2, var_3}
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = 4
    var_10 = [var_2, var_3, var_9]
    var_11 = module_0.pset(var_10)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 12/20 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = 4
    var_11 = [var_2, var_3, var_4, var_10]
    var_12 = 'c'
    var_13 = {var_6: var_2, var_7: var_3, var_12: var_4}
    var_14 = module_0.pmap(var_13)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 'value1'
    var_5 = 'arg1'
    var_6 = 'kwargs'
    var_7 = [var_0, var_1, var_3]
    var_8 = 'key1'
    var_9 = 'arg2'
    var_10 = {var_8: var_4, var_9: var_3}
    var_11 = module_0.pmap(var_10)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'set'
    var_3 = 1
    var_4 = [var_3]
    var_5 = 'a'
    var_6 = {var_5: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_3]
    var_9 = module_1.pset(var_8)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_non_empty_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 18/21 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, (2, 3)))
    assert var_6 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = module_0.freeze(var_12)
    var_14 = [var_3, var_4]
    var_15 = (var_6, var_7)
    var_16 = {var_9, var_10}
    var_17 = module_1.pset(var_16)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #10
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 13
    var_3 = 14
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/15 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = {var_0, var_1, var_3}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'nested'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_value'
    var_1 = 'existing_key'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_mutant_decorator_returns_callable.


def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 8/11 statements.
# Partially parsed test_mutant_decorator_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 7/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/7 statements.
# Failed to parse test_mutant_decorator_strict_false.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 'values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = bool(var_5 == {'values': [1, 2, 3]})
    assert var_8 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = {var_1, var_3}
    var_5 = [var_0, var_1, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 9/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = {var_5: var_0, var_7: var_1}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 'values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = bool(var_5 == {'values': [1, 2, 3]})
    assert var_8 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = {var_1, var_3}
    var_5 = [var_0, var_1, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = {var_2: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 7/11 statements.
# Partially parsed test_mutant_decorator_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 12/18 statements.
# Partially parsed test_mutant_decorator_no_mutation. Retrieved 7/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]

def test_case_0():
    var_0 = 'values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 5
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = bool(var_5 == {'values': [1, 2, 3]})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'extra'
    var_9 = [var_0, var_1]
    var_10 = {var_3: var_4}
    var_11 = module_0.pmap(var_10)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 13/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_pset_input. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_2, var_3, var_4}
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 4
    var_9 = [var_2, var_3, var_4, var_8]
    var_10 = 5
    var_11 = {var_2, var_3, var_4, var_10}
    var_12 = module_0.pset(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 42
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'new_key'
    var_7 = 'new_value'
    var_8 = {var_0: var_2, var_1: var_3, var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = {var_0, var_1, var_2, var_5}
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_set_and_list. Retrieved 12/20 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 42
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_4 == {'nested': {'value': 10}})
    assert var_10 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'values'
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = {var_2, var_3, var_5}
    var_9 = module_0.pset(var_8)
    var_10 = 4
    var_11 = [var_2, var_3, var_5, var_10]
    var_12 = bool(var_7 == {'items': {1, 2}, 'values': [1, 2, 3]})
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'extra'
    var_9 = [var_1]
    var_10 = {var_3: var_4}
    var_11 = module_0.pmap(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'config'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'enabled'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = bool(not (not (1, 2) and 0))
    assert var_0 is True



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_mutant_predicate.




# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 1/4 statements.
# Partially parsed test_mutant_with_pvector_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_pset_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = [var_4]
    var_6 = {var_1: var_0}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_4, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 'value'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 10
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'new'
    var_5 = 'value'
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = {var_0, var_1, var_2, var_5}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #28
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 16/27 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 'values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 5
    var_7 = [var_1, var_2, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = 'd'
    var_12 = [var_0]
    var_13 = [var_2]
    var_14 = [var_4]
    var_15 = [var_6]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = {var_1, var_3}
    var_5 = [var_0, var_1, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'value'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = {var_3, var_4}
    var_7 = 3
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_9 = [var_3, var_4, var_7]
    var_10 = [var_3, var_4, var_7]
    var_11 = module_0.pset(var_10)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_list_with_pvector. Retrieved 4/7 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/8 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 17/18 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

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
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, (2, 3)))
    assert var_6 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_1, var_2]
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_0, var_5, var_8]
    var_10 = module_0.freeze(var_9)
    var_11 = (var_2, var_3)
    var_12 = {var_1: var_11}
    var_13 = module_1.pmap(var_12)
    var_14 = {var_6, var_7}
    var_15 = module_2.pset(var_14)
    var_16 = [var_0, var_13, var_15]



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 11/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 'x'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_1, var_3]
    var_8 = 'key'
    var_9 = {var_4: var_5, var_8: var_3}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 'value'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_0: var_4, var_1: var_8}
    var_10 = 42
    var_11 = {var_2: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 99
    var_14 = [var_5, var_6, var_7, var_13]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_1, var_2, var_4, var_5]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 42
    var_1 = 'string'
    var_2 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 10/16 statements.
# Partially parsed test_mutant_with_no_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = 'result'
    var_6 = {var_1: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_0, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_1, var_2, var_4, var_5]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_2}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_4, var_5]

def test_case_0():
    var_0 = 'default'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_4: var_0}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_mutant_decorator_returns_callable.


def test_case_0():
    pass



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_freeze_with_defaultdict. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 12/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 'value'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 42
    var_8 = {var_3: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.pmap(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = 'd'
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_pset_input. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'values'
    var_1 = 'factor'
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]
    var_5 = 5
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 'inner'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = (var_0, var_1)



# Parsed testcases at query #41
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



