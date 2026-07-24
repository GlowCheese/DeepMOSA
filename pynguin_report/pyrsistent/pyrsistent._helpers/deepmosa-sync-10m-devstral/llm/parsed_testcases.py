####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/7 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict. Retrieved 9/10 statements.


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
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = []

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = []

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
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

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
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = {var_1: var_2}
    var_8 = [var_0, var_7]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_mutant_with_simple_types. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_list_and_dict. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_set. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/16 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = bool(var_5 == {'list': [1, 2, 3]})
    assert var_8 is True

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
    var_9 = bool(var_4 == {'nested': {'value': 5}})
    assert var_9 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'set'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = module_0.pset(var_7)
    var_9 = {var_0: var_8}
    var_10 = bool(var_5 == {'set': {1, 2, 3}})
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = bool(var_3 == (1, 2, 3))
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'kwargs'
    var_7 = 'x'
    var_8 = 'y'
    var_9 = {var_7: var_2, var_8: var_3}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_4: var_0, var_5: var_1, var_6: var_10}
    var_12 = module_0.pmap(var_11)

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
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = []
    var_7 = module_1.pset(var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_predicate_false. Retrieved 3/7 statements.


import builtins as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.type(*var_1, **var_2)
    var_4 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_predicate_false. Retrieved 3/7 statements.


import builtins as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.type(*var_1, **var_2)
    var_4 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/16 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 5/10 statements.


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
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4

def test_case_0():
    var_0 = 'lst'
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = [var_0, var_1, var_2, var_0]



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 19/35 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.


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
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

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
    var_7 = (var_0, var_1, var_2)
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new_key'
    var_11 = 'new_value'
    var_12 = {var_4: var_0, var_10: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = 5
    var_15 = {var_0, var_1, var_2, var_14}
    var_16 = module_1.pset(var_15)
    var_17 = (var_0, var_1, var_2)
    var_18 = 0

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
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'nested_list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 18/34 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/13 statements.


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
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

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
    var_7 = (var_0, var_1, var_2)
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new_key'
    var_11 = 'new_value'
    var_12 = {var_4: var_0, var_10: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_0, var_1, var_2, var_8}
    var_15 = module_1.pset(var_14)
    var_16 = (var_0, var_1, var_2)
    var_17 = 0

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
    var_0 = 'nested_list'
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
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 8/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/21 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 10/19 statements.
# Partially parsed test_mutant_decorator_returns_non_container. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 6/14 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 8/12 statements.


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
    var_0 = 1
    var_1 = 'nested'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = 'result'
    var_8 = [var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = [var_3, var_4]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = [var_0, var_1]
    var_5 = 0

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = {var_0, var_1}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5, var_0, var_1}
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_with_list_arg. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/27 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 19/33 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 13/27 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/13 statements.


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
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

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

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_2}
    var_8 = {var_0: var_5, var_1: var_7}
    var_9 = 4
    var_10 = [var_2, var_3, var_4, var_9]
    var_11 = 'new_key'
    var_12 = 'new_value'
    var_13 = {var_6: var_2, var_11: var_12}
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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'value1'
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'kwarg1'
    var_14 = 'kwarg_key'
    var_15 = 'kwarg_value'
    var_16 = {var_13: var_6, var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 1
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = {var_3}
    var_10 = module_1.pset(var_9)
    var_11 = 0
    var_12 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_mutable_return. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 10/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_8: var_1}
    var_10 = {var_6: var_0, var_8: var_1}
    var_11 = module_0.pmap(var_10)

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = {var_5: var_0}
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/7 statements.


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
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

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
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

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
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = {var_1: var_2}
    var_7 = [var_0, var_6]

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pset(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = bool(var_6 == (1, [2, 3]))
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.pmap(var_2)

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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector_with_elements. Retrieved 6/9 statements.


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
    var_4 = True
    var_5 = [var_4, var_1, var_2]

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
    var_8 = {var_0: var_6, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = False
    var_2 = module_0.freeze(var_0, var_1)
    assert var_2 == 42



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_with_list_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple_arguments. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_arguments. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 14/23 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = (var_0,)
    var_5 = {var_0}
    var_6 = 2
    var_7 = [var_0, var_6]
    var_8 = 'new_key'
    var_9 = {var_2: var_0, var_8: var_6}
    var_10 = module_0.pmap(var_9)
    var_11 = (var_0, var_6)
    var_12 = [var_0, var_6]
    var_13 = module_1.pset(var_12)

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
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'modified'
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

def test_case_0():
    var_0 = 42
    var_1 = 'string'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_mutant_predicate.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 16/17 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_pvector_with_strict_true. Retrieved 7/12 statements.
# Partially parsed test_freeze_pvector_with_strict_false. Retrieved 6/8 statements.


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
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = (var_2, var_5)
    var_7 = {var_1: var_6}
    var_8 = [var_0, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = {var_3, var_4}
    var_11 = module_1.pset(var_10)
    var_12 = (var_2, var_11)
    var_13 = {var_1: var_12}
    var_14 = module_2.pmap(var_13)
    var_15 = [var_0, var_14]

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.


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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 9/12 statements.


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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_func. Retrieved 1/3 statements.
# Partially parsed test_mutant_decorator_preserves_original_function. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_freeze_with_strict_true_and_pmap_input. Retrieved 15/18 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

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
    var_9 = module_0.pmap(var_8)
    var_10 = True
    var_11 = module_1.freeze(var_9, var_10)
    var_12 = [var_10, var_3]
    var_13 = {var_5: var_6}
    var_14 = module_0.pmap(var_13)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/19 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 7/12 statements.


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

def test_case_0():
    var_0 = 'lst'
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
    var_3 = 4
    var_4 = 5
    var_5 = [var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 'd'
    var_10 = [var_3, var_4]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = bool(not (not False and True))
    assert var_0 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_simple_immutable_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 10/20 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 18/32 statements.
# Partially parsed test_mutant_with_pyrsistent_types. Retrieved 16/24 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

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
    var_9 = 0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 4

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'tuple'
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = {var_9, var_10}
    var_12 = 6
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = (var_12, var_15)
    var_17 = {var_0: var_8, var_1: var_11, var_2: var_16}

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = module_1.pset(var_9)
    var_11 = [var_0, var_1]
    var_12 = {var_3: var_4}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_7, var_8}
    var_15 = module_1.pset(var_14)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_pvector_with_strict. Retrieved 6/9 statements.
# Partially parsed test_freeze_pvector_without_strict. Retrieved 5/7 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 17/22 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_0: var_6, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

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

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = {var_7, var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_0, var_5, var_10]
    var_12 = [var_2, var_3]
    var_13 = [var_7, var_8]
    var_14 = module_0.pset(var_13)
    var_15 = (var_6, var_14)
    var_16 = module_1.freeze(var_11)



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/16 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_2]
    var_4 = 'x'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

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
    var_5 = 5
    var_6 = [var_0, var_1, var_2, var_4, var_5]
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_mixed. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict. Retrieved 12/13 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/7 statements.


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
    var_2 = []
    var_3 = module_1.pset(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

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

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_mutant_predicate.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_non_strict. Retrieved 8/11 statements.


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

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]

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



# Parsed testcases at query #42
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.freeze(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_simple_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_pvector_arg. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_arg. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_pset_arg. Retrieved 8/11 statements.


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
    var_0 = 'inner_list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]
    var_8 = bool(var_5 == {'inner_list': [1, 2, 3]})
    assert var_8 is True

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



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #45
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_freeze_predicate_false. Retrieved 4/8 statements.


import builtins as module_0

def test_case_0():
    var_0 = {}
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.type(*var_1, **var_2)
    var_4 = True
    var_5 = {}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_with_non_empty_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_mixed_types. Retrieved 19/20 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_with_defaultdict. Retrieved 6/9 statements.
# Partially parsed test_freeze_with_non_strict_mode. Retrieved 8/11 statements.


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
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, (2, 3)))
    assert var_6 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = set(var_9)
    var_11 = [var_0, var_3, var_6, var_10]
    var_12 = module_0.freeze(var_11)
    var_13 = {var_1: var_2}
    var_14 = module_1.pmap(var_13)
    var_15 = (var_4, var_5)
    var_16 = [var_7, var_8]
    var_17 = module_2.pset(var_16)
    var_18 = [var_0, var_14, var_15, var_17]

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_0: var_6, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_1: var_2}
    var_9 = {var_0: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(var_7 == var_10)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_nested_list. Retrieved 8/9 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 8/11 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 7/10 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/7 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 10/16 statements.


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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_1.pmap(var_5)
    var_7 = [var_6]

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_2,)
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_1.pmap(var_5)
    var_7 = (var_6,)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_1.freeze(var_5)
    var_7 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = False
    var_5 = {var_0: var_1}
    var_6 = [var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.pmap(var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = {var_1: var_5}
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_3]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_with_strict_true_converts_pmap. Retrieved 6/7 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_non_strict. Retrieved 8/11 statements.


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

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 14/19 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 5/11 statements.


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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'x'
    var_5 = 'y'
    var_6 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 10
    var_6 = {var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_with_non_empty_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_mixed_containers. Retrieved 11/17 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_with_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_non_strict_mode. Retrieved 8/11 statements.


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
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, (2, 3)))
    assert var_6 is True

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
    var_4 = True
    var_5 = [var_4, var_1, var_2]

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
    var_8 = {var_0: var_6, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)
    var_8 = bool(var_5 == var_7)
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_set_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'list'
    var_1 = 'nested_dict'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = {var_0: var_5, var_1: var_8}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 3/6 statements.


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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_1, var_3, var_4]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_values. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 13/17 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/8 statements.
# Partially parsed test_freeze_nested_defaultdict. Retrieved 8/13 statements.


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

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = [var_5]
    var_7 = (var_4, var_6)
    var_8 = [var_0, var_3, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = {var_1: var_2}
    var_11 = module_1.pmap(var_10)
    var_12 = [var_5]

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = [var_3, var_1]

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_8: var_1}
    var_10 = {var_6: var_0, var_8: var_1}
    var_11 = module_0.pmap(var_10)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_mutant_returns_frozen. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_pset. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_pmap. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2

def test_case_0():
    var_0 = 'inner'
    var_1 = 'value'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/29 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 14/26 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 15/24 statements.
# Partially parsed test_mutant_with_no_freeze_needed. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 12/16 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 9/15 statements.


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
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = 'result'
    var_6 = [var_1, var_2]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_0, var_1]
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = [var_0, var_1]

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
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = {var_7: var_10}
    var_12 = 'x'
    var_13 = [var_8, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_5, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'kwargs'
    var_11 = [var_1, var_2]
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_5, var_6]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_2, var_3, var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = {var_0, var_1}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5, var_0, var_1}
    var_7 = module_0.pset(var_6)
    var_8 = {var_0, var_1}
    var_9 = module_0.pset(var_8)
    var_10 = {var_3, var_4, var_5, var_0, var_1}
    var_11 = module_0.pset(var_10)

def test_case_0():
    var_0 = 3
    var_1 = 4
    var_2 = (var_0, var_1)
    var_3 = 5
    var_4 = [var_1, var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_1, var_3]
    var_7 = 1
    var_8 = 2



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = bool(not (not False and True))
    assert var_0 is True



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 9/12 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 19/29 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_non_container_types. Retrieved 3/8 statements.


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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = 5
    var_7 = 6
    var_8 = [var_7]
    var_9 = (var_6, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'kwargs'
    var_13 = [var_1]
    var_14 = 'c'
    var_15 = 'd'
    var_16 = {var_3, var_4}
    var_17 = module_0.pset(var_16)
    var_18 = [var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = [var_0, var_1, var_2]

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
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_3 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = []
    var_3 = set()
    var_4 = module_1.pset(var_3)
    var_5 = ()

def test_case_0():
    var_0 = 42
    var_1 = 'string'
    var_2 = None



