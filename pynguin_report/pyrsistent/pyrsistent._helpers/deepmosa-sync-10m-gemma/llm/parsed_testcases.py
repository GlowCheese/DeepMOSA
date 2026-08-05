####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 18/26 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_15 = [var_10, var_11]
    var_16 = module_0.pset(var_15)
    var_17 = module_1.freeze(var_13)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.freeze(var_0)
    assert var_1 == 10
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_1]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #2
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.freeze(var_0)
    assert var_1 == 5



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_list_to_pvector. Retrieved 9/13 statements.
# Partially parsed test_freeze_dict_to_pmap. Retrieved 15/19 statements.
# Partially parsed test_freeze_tuple. Retrieved 8/11 statements.
# Partially parsed test_freeze_set_to_pset. Retrieved 8/9 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 15/21 statements.
# Partially parsed test_freeze_strict_false_on_dict. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_containers. Retrieved 10/11 statements.


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
    var_5 = [var_1]
    var_6 = [var_0, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = [var_1, var_5]
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_6, var_4: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = [var_1, var_5]
    var_13 = {var_7: var_8}
    var_14 = module_1.pmap(var_13)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]
    var_6 = (var_0, var_1)
    var_7 = module_0.freeze(var_6)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0, var_1}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_1.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_12 = [var_7, var_8]
    var_13 = module_0.pset(var_12)
    var_14 = module_1.freeze(var_10)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'a'
    var_5 = [var_0]
    var_6 = {var_4: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_0.freeze(var_2)
    var_4 = module_1.pmap()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True
    var_6 = ()
    var_7 = module_0.freeze(var_6)
    var_8 = bool(var_7 == ())
    assert var_8 is True
    var_9 = set()
    var_10 = module_0.freeze(var_9)
    var_11 = module_2.pset()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_strict_pmap_is_true. Retrieved 10/13 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1
import builtins as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_2.type(*var_8, **var_9)
    var_11 = isinstance(var_5, var_10)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_strict_pmap_returns_pmap. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 11/19 statements.
# Partially parsed test_mutant_preserves_logic_while_freezing. Retrieved 4/9 statements.
# Partially parsed test_mutant_deep_freeze. Retrieved 7/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'a'
    var_8 = [var_0, var_1, var_2]
    var_9 = {var_4: var_5}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]

def test_case_0():
    var_0 = 'outer'
    var_1 = 1
    var_2 = 'inner'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/15 statements.
# Partially parsed test_mutant_isolates_mutation. Retrieved 6/13 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/16 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

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
    var_0 = 'key'
    var_1 = 1
    var_2 = 'inner'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 3/10 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_preserves_simple_values. Retrieved 3/11 statements.
# Partially parsed test_mutant_deep_freeze_of_kwargs. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 10
    var_1 = 'string'
    var_2 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'item'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_dict_triggers_line_32_logic. Retrieved 7/12 statements.


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
    var_9 = bool(var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.pmap()
    var_6 = var_5.__class__



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'original'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'mutated'
    var_4 = 'mutated'
    var_5 = bool('mutated' not in var_2)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 10/17 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 3/10 statements.
# Partially parsed test_mutant_handles_simple_types. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = {var_4: var_5}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 10
    var_1 = 'string'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_list_simple. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_nested. Retrieved 7/10 statements.
# Partially parsed test_freeze_dict_nested. Retrieved 13/16 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 7/9 statements.
# Partially parsed test_freeze_dict_with_defaultdict. Retrieved 5/11 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 28/42 statements.
# Partially parsed test_freeze_already_frozen. Retrieved 10/13 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.freeze(var_0)
    assert var_1 == 123

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

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
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = [var_2, var_3]
    var_11 = {var_5: var_6}
    var_12 = module_1.pmap(var_11)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == (1, 2))
    assert var_4 is True

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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

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
    var_2 = 2
    var_3 = [var_2]
    var_4 = (var_1, var_3)
    var_5 = {var_0: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = [var_5, var_10]
    var_12 = [var_2]
    var_13 = '4'
    var_14 = {var_13: var_8}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_6, var_15]
    var_17 = [var_2]
    var_18 = (var_1, var_17)
    var_19 = {var_0: var_18}
    var_20 = {var_7: var_8}
    var_21 = [var_6, var_20]
    var_22 = [var_19, var_21]
    var_23 = [var_2]
    var_24 = {var_7: var_8}
    var_25 = module_0.pmap(var_24)
    var_26 = [var_6, var_25]
    var_27 = module_1.freeze(var_22)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.pmap(var_5)
    var_7 = module_1.freeze(var_6)
    var_8 = {var_4: var_0}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_mutant_predicate_is_false.
# Partially parsed test_mutant_functionality. Retrieved 8/16 statements.


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
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_list. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_set. Retrieved 4/6 statements.
# Partially parsed test_freeze_recursive_list. Retrieved 7/11 statements.


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
    var_4 = bool(var_3 == (1, 2))
    assert var_4 is True

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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = module_1.freeze(var_4)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = [var_1, var_2]
    var_6 = module_0.freeze(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_handles_simple_list. Retrieved 6/7 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.freeze(var_0)
    assert var_1 == 5

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.freeze(var_0)
    assert var_1 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_strict_pmap_returns_pmap. Retrieved 5/7 statements.
# Partially parsed test_freeze_dict_returns_pmap. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 14/24 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 3/12 statements.
# Partially parsed test_mutant_preserves_functionality. Retrieved 4/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'inner'
    var_8 = [var_0]
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_1, var_2]
    var_11 = {var_4: var_5}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_0]

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 19/22 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_4,)
    var_6 = {var_3: var_5}
    var_7 = [var_2, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = (var_4,)
    var_13 = {var_3: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8, var_9]
    var_17 = module_1.pset(var_16)
    var_18 = module_2.freeze(var_11)

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_predicate_evaluates_to_false. Retrieved 2/8 statements.
# Partially parsed test_mutant_freezing_behavior. Retrieved 5/11 statements.
# Partially parsed test_mutant_kwargs_freezing. Retrieved 5/11 statements.


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'a'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 7/16 statements.
# Partially parsed test_mutant_behavior_with_lists. Retrieved 6/14 statements.
# Partially parsed test_mutant_behavior_with_dicts. Retrieved 3/10 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 1/6 statements.
# Partially parsed test_mutant_handles_kwargs_freezing. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

def test_case_0():
    var_0 = 'old'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'c'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'b'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 8/14 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_preserves_simple_types. Retrieved 3/9 statements.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 'b'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = [var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 10



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_is_decorator. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = 0
    var_6 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.type(*var_4, **var_5)



# Parsed testcases at query #27
#--------------------------




import pyrsistent._helpers as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.type(*var_4, **var_5)



# Parsed testcases at query #28
#--------------------------




import pyrsistent._helpers as module_0
import builtins as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.type(*var_2, **var_3)
    var_5 = {}
    var_6 = module_2.pmap(var_5)
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = bool(var_4 is var_9)
    assert var_10 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_input_arguments. Retrieved 3/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_freezes_input_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 7/15 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 4/12 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_2 == {'key': 'original'})
    assert var_6 is True

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
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = [var_0]
    var_4 = bool(var_2 == [[1]])
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 'hello'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_decorator_identity. Retrieved 11/18 statements.
# Partially parsed test_mutant_decorator_freezes_inputs. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 8/14 statements.


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
    var_7 = {var_0: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_1, var_4]
    var_10 = module_1.pset(var_9)

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
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 4/12 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = [var_0]

def test_case_0():
    var_0 = 5



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'b'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 10/19 statements.
# Partially parsed test_mutant_preserves_logic_with_nested_structures. Retrieved 6/11 statements.
# Partially parsed test_mutant_handles_kwargs. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_primitive_types. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = 'a'
    var_9 = 'b'

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'b'

def test_case_0():
    var_0 = 5
    var_1 = 'string'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/10 statements.
# Partially parsed test_mutant_recursive_freezing. Retrieved 9/21 statements.
# Partially parsed test_mutant_kwargs_freezing. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = [var_5]
    var_7 = (var_4, var_6)
    var_8 = [var_0, var_3, var_7]

def test_case_0():
    var_0 = 'value'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #36
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.freeze(var_0)
    assert var_1 == 10



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/16 statements.
# Partially parsed test_mutant_isolates_mutation_in_args. Retrieved 6/15 statements.
# Partially parsed test_mutant_handles_complex_nested_structures. Retrieved 11/21 statements.
# Partially parsed test_mutant_handles_kwargs_mutation. Retrieved 5/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

def test_case_0():
    var_0 = 'key'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 'inner'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = {var_0: var_6, var_1: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 19/22 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_4,)
    var_6 = {var_3: var_5}
    var_7 = [var_2, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = (var_4,)
    var_13 = {var_3: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8, var_9]
    var_17 = module_1.pset(var_16)
    var_18 = module_2.freeze(var_11)

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
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #39
#--------------------------




import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_is_decorator. Retrieved 13/18 statements.


import pyrsistent._helpers as module_0
import builtins as module_1

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
    var_9 = module_0.freeze(var_8)
    var_10 = {var_0: var_2}
    var_11 = module_0.freeze(var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.type(*var_12, **var_13)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 6/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0.pmap(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_mutant_freezes_arguments.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/15 statements.
# Partially parsed test_mutant_prevents_mutation_of_inputs. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/12 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 1/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_0, var_1, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = bool(var_2 == [1, 2])
    assert var_3 is True
    var_4 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = []



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.


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
    var_2 = {var_0, var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0, var_1}
    var_5 = module_1.pset(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 18/26 statements.
# Partially parsed test_freeze_non_recursive_keys. Retrieved 7/8 statements.
# Partially parsed test_freeze_strict_false_behavior. Retrieved 9/10 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_15 = [var_10, var_11]
    var_16 = module_0.pset(var_15)
    var_17 = module_1.freeze(var_13)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = var_6[1, [2]]
    assert var_7 == 3

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
    var_8 = var_7[var_0]

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 13/17 statements.
# Partially parsed test_freeze_strict_false_on_pmap. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_keys_not_frozen. Retrieved 6/9 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'x'
    var_4 = 4
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = (var_1, var_6)
    var_8 = [var_0, var_7]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_2, var_10]
    var_12 = module_1.freeze(var_8)

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
    var_8 = bool(var_7 == var_5)
    assert var_8 is True
    var_9 = var_7[var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.freeze(var_0)
    assert var_1 == 10
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = (var_7,)
    var_9 = [var_6, var_8]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 19/22 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 4/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/11 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_4,)
    var_6 = {var_3: var_5}
    var_7 = [var_2, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = module_0.freeze(var_11)
    var_13 = (var_4,)
    var_14 = {var_3: var_13}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_2, var_15]
    var_17 = [var_8, var_9]
    var_18 = module_2.pset(var_17)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = False

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 11/24 statements.
# Partially parsed test_mutant_with_empty_inputs. Retrieved 1/7 statements.
# Partially parsed test_mutant_preserves_non_mutable_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = [var_0, var_5, var_8]
    var_10 = 0

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 10



# Parsed testcases at query #4
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == 3.14)
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.freeze(var_0)
    assert var_1 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.freeze(var_0)
    assert var_1 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == 3.14)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.thaw(var_0)
    assert var_1 == 'hello'

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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = module_1.thaw(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'c'
    var_4 = {var_3: var_2}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_1, var_7: var_5}
    var_9 = module_0.m(**var_8)
    var_10 = [var_0, var_9]
    var_11 = module_1.v(*var_10)
    var_12 = module_2.thaw(var_11)
    var_13 = bool(var_12 == [1, {'a': 2, 'b': {'c': 3}}])
    assert var_13 is True

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.v(*var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.v(*var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.v(*var_6)
    var_8 = module_1.thaw(var_7)
    var_9 = bool(var_8 == [[1], [2]])
    assert var_9 is True

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

import pyrsistent._pvector as module_0
import pyrsistent._pmap as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.v(*var_2)
    var_4 = 3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_1.m(**var_7)
    var_9 = module_2.thaw(var_8)
    var_10 = bool(var_9 == {'a': [1, 2], 'b': 3})
    assert var_10 is True

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = 4
    var_6 = (var_5,)
    var_7 = (var_0, var_4, var_6)
    var_8 = module_1.thaw(var_7)
    var_9 = bool(var_8 == (1, [2, 3], (4,)))
    assert var_9 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.thaw(var_4)
    var_6 = bool(var_5 == [1, [2, 3]])
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.thaw(var_4)
    var_6 = bool(var_5 == {'a': {'b': 1}})
    assert var_6 is True

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.v(*var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = module_1.thaw(var_4, var_5)
    var_7 = [var_0, var_1]
    var_8 = module_0.v(*var_7)
    var_9 = [var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = {var_0: var_4}
    var_6 = False
    var_7 = module_1.thaw(var_5, var_6)
    var_8 = 'b'
    var_9 = {var_8: var_1}
    var_10 = module_0.m(**var_9)
    var_11 = {var_0: var_10}
    var_12 = bool(var_7 == var_11)
    assert var_12 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_strict_pmap_true. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 6/8 statements.
# Partially parsed test_freeze_list_of_dicts. Retrieved 13/14 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 9/15 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.freeze(var_0)
    assert var_1 == 123

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

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
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == (1, 2))
    assert var_4 is True

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_2 = 2
    var_3 = [var_2]
    var_4 = (var_1, var_3)
    var_5 = {var_0: var_4}
    var_6 = [var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_1]
    var_7 = {var_0: var_6}
    var_8 = module_1.pmap(var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_is_recursive. Retrieved 12/25 statements.
# Failed to parse test_mutant_handles_empty_inputs.
# Partially parsed test_mutant_preserves_non_container_types. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 'inner'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = {var_0: var_6, var_1: var_10}

def test_case_0():
    var_0 = 10
    var_1 = 'hello'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_strict_pmap_is_true. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_strict_pmap_returns_pmap. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = var_5['a']
    assert var_6 == 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 2/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = var_0.__class__



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 10/19 statements.


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = '__len__'
    var_12 = bool(True)
    assert var_12 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/14 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 3/9 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 9/19 statements.
# Partially parsed test_mutant_with_no_mutation_needed. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2['a']
    assert var_5 == 0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5
    var_1 = 'string'



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_mutant_decorator_returns_function.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_returns_frozen_value. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_freezes_args. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 3/8 statements.
# Failed to parse test_mutant_is_not_none_and_is_callable.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 6/14 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 6/21 statements.
# Partially parsed test_mutant_handles_primitives. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_0]
    var_6 = {var_2: var_3}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 'new_key'
    var_4 = 2
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 8/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap()
    var_7 = var_6.__class__



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 1/7 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_ensures_return_value_is_frozen. Retrieved 1/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]

def test_case_0():
    var_0 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_is_decorator. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = 0
    var_6 = 'b'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 11/23 statements.
# Partially parsed test_mutant_preserves_primitives. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 2
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = {var_1: var_5}
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = [var_3]

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 8/15 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_handles_complex_nesting. Retrieved 3/15 statements.
# Partially parsed test_mutant_preserves_functionality. Retrieved 4/11 statements.
# Partially parsed test_mutant_kwargs_are_frozen. Retrieved 7/17 statements.


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
    var_0 = 0
    var_1 = 1
    var_2 = 'c'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'c'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'b'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 10/17 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 13/26 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = {var_4: var_5}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 'inner'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = 5
    var_9 = 6
    var_10 = {var_8, var_9}
    var_11 = [var_4, var_7, var_10]
    var_12 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_preserves_functionality. Retrieved 10/14 statements.


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
    var_7 = 'b'
    var_8 = {var_7: var_4}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_deep_freeze_nesting. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 10
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = [var_4]
    var_6 = (var_0, var_5)
    var_7 = 'nested'
    var_8 = [var_0]
    var_9 = {var_7: var_8}
    var_10 = [var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_decorator_evaluates_to_true. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'c'
    var_8 = {var_7: var_5}
    var_9 = [var_2, var_3, var_8]
    var_10 = 'immutable'
    var_11 = var_9[var_3]
    var_12 = var_9[var_3]
    var_13 = hasattr(var_12, var_10)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluation. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 2/11 statements.
# Partially parsed test_mutant_recursive_freezing. Retrieved 9/18 statements.
# Partially parsed test_mutant_handles_primitive_types. Retrieved 3/9 statements.


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
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 'deep'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = (var_1, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = 0

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/11 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 6/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 19/23 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.freeze(var_0, var_1)
    var_3 = {}
    var_4 = module_1.pmap(var_3)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

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
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0, var_1}
    var_5 = module_1.pset(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_4,)
    var_6 = {var_3: var_5}
    var_7 = [var_2, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = (var_4,)
    var_13 = {var_3: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_2, var_14]
    var_16 = {var_8, var_9}
    var_17 = module_1.pset(var_16)
    var_18 = module_2.freeze(var_11)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == {[1]})
    assert var_4 is True
    var_5 = 2
    var_6 = (var_0, var_5)
    var_7 = {var_6}
    var_8 = module_0.freeze(var_7)
    var_9 = (var_0, var_5)
    var_10 = {var_9}
    var_11 = module_1.pset(var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

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
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality_and_freezes_inputs. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 12/23 statements.
# Partially parsed test_mutant_preserves_functionality_with_mutation_inside. Retrieved 6/12 statements.
# Partially parsed test_mutant_handles_primitive_types. Retrieved 3/9 statements.
# Partially parsed test_mutant_recursive_freezing_of_kwargs. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = 'key'
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = [var_2]
    var_11 = [var_6, var_7]

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = 'nested'
    var_2 = 2
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = [var_0, var_4]
    var_6 = 'outer'
    var_7 = 3
    var_8 = [var_7]
    var_9 = {var_6: var_8}
    var_10 = [var_2]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 9/24 statements.


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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 9/16 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_deep_freeze_on_complex_structure. Retrieved 4/15 statements.


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
    var_0 = 1

def test_case_0():
    var_0 = 0
    var_1 = 'key'
    var_2 = 3
    var_3 = []



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 12/19 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 7/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 3/10 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 4/10 statements.


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
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_2, var_3]
    var_10 = {var_5: var_6}
    var_11 = module_0.pmap(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = {var_1: var_2}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 2
    var_3 = (var_0, var_2)



# Parsed testcases at query #39
#--------------------------




import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1
import builtins as module_2

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
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_2.type(*var_9, **var_10)
    var_12 = module_0.pmap()
    var_13 = var_12.__class__
    var_14 = var_11 is var_13
    var_15 = 'items'
    var_16 = hasattr(var_5, var_15)
    var_17 = var_14 or var_16
    var_18 = bool(var_8 and var_17)
    assert var_18 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_list. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_tuple. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_set. Retrieved 4/6 statements.


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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_is_decorator. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = 0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_mutant_decorator_returns_function. Retrieved 3/7 statements.
# Partially parsed test_mutant_decorator_execution. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_freezes_inputs. Retrieved 6/12 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = var_0[var_4]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true_at_line_32. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/11 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/10 statements.


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
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 2
    var_3 = 'inner'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = 4
    var_9 = (var_8,)
    var_10 = [var_0, var_7, var_9]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 14/24 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 6/15 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_0, var_1}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 17/31 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap()
    var_7 = var_6.__class__
    var_8 = 2
    var_9 = [var_1, var_8]
    var_10 = module_1.pset(var_9)
    var_11 = [var_1, var_8]
    var_12 = module_1.pset(var_11)
    var_13 = module_1.pset()
    var_14 = var_13.__class__
    var_15 = [var_1, var_8]
    var_16 = [var_1, var_8]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 16/28 statements.
# Partially parsed test_mutant_decorator_freezes_arguments. Retrieved 18/37 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 19/30 statements.


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
    var_7 = 3
    var_8 = [var_1, var_6, var_7]
    var_9 = module_1.pset(var_8)
    var_10 = [var_1, var_6, var_7]
    var_11 = module_1.pset(var_10)
    var_12 = [var_1, var_6]
    var_13 = [var_1, var_6]
    var_14 = [var_1, var_6]
    var_15 = [var_1, var_6]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import builtins as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.type(*var_15, **var_16)
    var_18 = module_1.pset()
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.type(*var_19, **var_20)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import builtins as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 'm'
    var_14 = module_0.pmap()
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.type(*var_15, **var_16)
    var_18 = 's'
    var_19 = module_1.pset()
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.type(*var_20, **var_21)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 2/9 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 5/12 statements.
# Partially parsed test_mutant_preserves_functionality. Retrieved 4/9 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 2
    var_4 = [var_3]
    var_5 = {var_2: var_4}

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 'outer'
    var_1 = 1
    var_2 = 'inner'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}



