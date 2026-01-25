####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 5/12 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_non_container_arguments. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 12/23 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 7/13 statements.
# Partially parsed test_mutant_decorator_with_defaultdict. Retrieved 8/17 statements.
# Partially parsed test_mutant_decorator_strict_false_behavior. Retrieved 6/14 statements.
# Partially parsed test_mutant_decorator_empty_arguments. Retrieved 2/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = 99
    var_11 = [var_10, var_3, var_4]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'extra'
    var_5 = 100
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 10
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'empty'
    var_1 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_int. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/9 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 8/11 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 10/13 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0, var_1]
    var_9 = [var_3, var_4]

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 6
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = [var_1, var_2, var_3]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

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
    var_0 = ()
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0, var_1]
    var_9 = [var_3, var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 4
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = [var_1, var_2]



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 12/29 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 100
    var_6 = module_0.m()
    var_7 = 999
    var_8 = {}
    var_9 = module_1.freeze(var_8)
    var_10 = set()
    var_11 = module_1.freeze(var_10)



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_with_recursive_structures. Retrieved 10/19 statements.
# Partially parsed test_mutant_decorator_with_frozen_inputs. Retrieved 5/10 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/9 statements.
# Partially parsed test_mutant_decorator_no_side_effects_on_multiple_calls. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = 1
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
    pass

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'inner'
    var_6 = [var_1, var_2]
    var_7 = 7
    var_8 = 8
    var_9 = [var_7, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

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
    var_3 = [var_0, var_1]

def test_case_0():
    var_0 = 0
    assert var_0 == 2
    var_1 = 10



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 7/13 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 12/24 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 8/14 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 15/27 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 1/5 statements.
# Partially parsed test_mutant_decorator_freezes_returned_mutable. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = {var_0, var_1, var_2, var_4}
    var_6 = module_0.pset(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 99
    var_8 = [var_0, var_1, var_7]
    var_9 = 100
    var_10 = {var_3: var_9}
    var_11 = module_0.pmap(var_10)

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
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 'inner'
    var_6 = 'old'
    var_7 = {var_5: var_6}
    var_8 = (var_4, var_7)
    var_9 = {var_0: var_3, var_1: var_8}
    var_10 = 'changed'
    var_11 = [var_10]
    var_12 = 'updated'
    var_13 = {var_5: var_12}
    var_14 = module_0.pmap(var_13)

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 14/27 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 11/26 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_decorator_with_strict_false. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new'
    var_11 = 'value'
    var_12 = {var_4: var_5, var_10: var_11}
    var_13 = module_0.pmap(var_12)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2, var_0]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = [var_2]
    var_6 = (var_4, var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'mutated'
    var_9 = [var_8]
    var_10 = [var_2, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_1, var_4, var_5]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_empty_dict. Retrieved 3/4 statements.
# Partially parsed test_freeze_dict_with_values. Retrieved 9/11 statements.
# Partially parsed test_freeze_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_set. Retrieved 5/6 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 2/9 statements.
# Partially parsed test_freeze_strict_false. Retrieved 8/10 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 9/11 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 6/11 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/22 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

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
    var_8 = var_7[var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)

def test_case_0():
    var_0 = 'a'
    var_1 = 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = var_6[var_0]

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
    var_8 = var_7[var_0]

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = module_0.freeze(var_11)
    var_13 = var_12[var_0]
    var_14 = var_12[var_0][var_2]
    var_15 = var_12[var_1]
    var_16 = var_12[var_1][var_2]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 9/22 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 3/10 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 1/10 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'original'
    var_6 = {var_4: var_5}
    var_7 = 'list'
    var_8 = 'dict'

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 3

def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = (var_4, var_7)
    var_9 = {var_0: var_3, var_1: var_8}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 4/15 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 8/17 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 1/8 statements.
# Partially parsed test_mutant_decorator_with_strict_freeze. Retrieved 6/13 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'

def test_case_0():
    var_0 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 13/23 statements.


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
    var_7 = 3
    var_8 = [var_1, var_4, var_7]
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = len(var_8)
    assert var_12 == 3



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 2/8 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 4/12 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 2/6 statements.
# Partially parsed test_mutant_strict_false_implicitly. Retrieved 2/9 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = 3

def test_case_0():
    var_0 = 'inner'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3

def test_case_0():
    var_0 = 5
    var_1 = (var_0,)

def test_case_0():
    var_0 = 10
    var_1 = [var_0]



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 14/27 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 12/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new'
    var_11 = 'value'
    var_12 = {var_4: var_5, var_10: var_11}
    var_13 = module_0.pmap(var_12)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'answer'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = {var_4, var_5}
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'mutated'
    var_9 = [var_8]
    var_10 = {var_4, var_5}
    var_11 = module_0.pset(var_10)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_non_container_arguments. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures_in_arguments. Retrieved 8/17 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 4/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
    var_4 = 'dict'
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 11/19 statements.
# Partially parsed test_mutant_return_value_frozen. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_mutables. Retrieved 13/22 statements.
# Partially parsed test_mutant_no_side_effects_on_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_empty_args. Retrieved 2/9 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'initial'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = 'factor'
    var_9 = {var_3: var_4, var_8: var_6}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_2, var_3, var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = 99
    var_9 = [var_8, var_3]
    var_10 = 4
    var_11 = {var_2, var_3, var_5, var_10}
    var_12 = module_0.pset(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 'empty'
    var_1 = []



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 2/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 2/7 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 2/9 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 4/19 statements.
# Failed to parse test_mutant_return_value_frozen.
# Partially parsed test_mutant_nested_structures. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = set()
    var_1 = set()

def test_case_0():
    var_0 = 0
    var_1 = (var_0,)

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = set()

def test_case_0():
    var_0 = 'list'
    var_1 = 'original'
    var_2 = [var_1]
    var_3 = {var_0: var_2}



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_mutable_list_input. Retrieved 8/10 statements.
# Partially parsed test_mutant_with_mutable_dict_input. Retrieved 12/13 statements.
# Partially parsed test_mutant_with_mutable_set_input. Retrieved 9/10 statements.
# Partially parsed test_mutant_with_nested_mutable_input. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 5/6 statements.
# Partially parsed test_mutant_returns_frozen_output. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 3/4 statements.
# Failed to parse test_mutant_preserves_function_name.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = module_0.mutant(var_0)
    var_2 = 3
    var_3 = 4

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda lst: lst.append(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5, var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = lambda d: d.update(var_2)
    var_4 = module_0.mutant(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_5: var_7, var_6: var_8, var_0: var_1}
    var_11 = module_1.pmap(var_10)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = lambda s: s.add(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = {var_3, var_4, var_5, var_0}
    var_8 = module_1.pset(var_7)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = lambda x: x[var_0].append(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = [var_6]
    var_8 = [var_4, var_5, var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0
    var_2 = module_0.mutant(var_1)



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_decorator_does_not_mutate_inputs. Retrieved 19/29 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.pset(var_8)
    var_10 = {var_5: var_0}
    var_11 = module_1.pmap(var_10)
    var_12 = 4
    var_13 = [var_0, var_1, var_2, var_12]
    var_14 = module_0.pset(var_13)
    var_15 = 'new'
    var_16 = 99
    var_17 = {var_5: var_0, var_15: var_16}
    var_18 = module_1.pmap(var_17)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Failed to parse test_mutant_decorator_with_non_mutable_return.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 13/27 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 5
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_0, var_2]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = (var_5, var_7)
    var_9 = {var_0: var_4, var_1: var_8}
    var_10 = 99
    var_11 = [var_2, var_3, var_10]
    var_12 = [var_6]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/24 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_frozen_inputs. Retrieved 14/25 statements.
# Partially parsed test_mutant_decorator_with_non_container_arguments. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_with_strict_false. Retrieved 1/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 5
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_0, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0]
    var_7 = {var_2: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_0, var_3]
    var_10 = 'b'
    var_11 = 2
    var_12 = {var_2: var_3, var_10: var_11}
    var_13 = module_0.pmap(var_12)

def test_case_0():
    var_0 = 5
    var_1 = 'hello'

def test_case_0():
    var_0 = {}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 10/15 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = module_0.freeze(var_5)
    var_7 = {var_1: var_2}
    var_8 = module_1.pmap(var_7)
    var_9 = [var_0, var_8, var_4]

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

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = [var_2, var_3]
    var_13 = [var_6, var_7]
    var_14 = module_1.pset(var_13)
    var_15 = {var_5: var_14}
    var_16 = module_2.pmap(var_15)

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_1: var_2}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_0, var_9]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_5]
    var_7 = True
    var_8 = module_1.freeze(var_6, var_7)
    var_9 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_non_container_arguments. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_with_strict_freeze. Retrieved 6/14 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 13/25 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs_dict_values. Retrieved 6/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 10
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = (var_4, var_7)
    var_9 = {var_0: var_3, var_1: var_8}
    var_10 = 'changed'
    var_11 = [var_10]
    var_12 = [var_5, var_6]

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
    var_2 = [var_0, var_1]
    var_3 = 'list'
    var_4 = 100
    var_5 = [var_4]



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------






# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_mutable_structures. Retrieved 14/22 statements.
# Failed to parse test_mutant_preserves_function_metadata.
# Partially parsed test_mutant_with_empty_arguments. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/13 statements.


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
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

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
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_2, var_3, var_6]
    var_10 = 'd'
    var_11 = 4
    var_12 = {var_5: var_6, var_10: var_11}
    var_13 = module_0.pmap(var_12)

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = (var_2,)
    var_4 = [var_0, var_1]
    var_5 = 3



# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 13/27 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = (var_4, var_7)
    var_9 = {var_0: var_3, var_1: var_8}
    var_10 = 'changed'
    var_11 = [var_10]
    var_12 = [var_5, var_6]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 16/19 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false. Retrieved 9/10 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 9/12 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
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

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = (var_5, var_8)
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = [var_2, var_3]
    var_13 = {var_6, var_7}
    var_14 = module_1.pset(var_13)
    var_15 = (var_5, var_14)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = {var_1: var_2}
    var_8 = [var_0, var_7]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
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
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = {var_1: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_5, var_7]

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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_thaw_pvector. Retrieved 4/7 statements.
# Partially parsed test_thaw_pvector_nested. Retrieved 6/9 statements.
# Partially parsed test_thaw_pmap_nested. Retrieved 5/9 statements.
# Partially parsed test_thaw_pset. Retrieved 4/7 statements.
# Partially parsed test_thaw_tuple_nested. Retrieved 5/9 statements.
# Partially parsed test_thaw_strict_false_list. Retrieved 5/8 statements.
# Partially parsed test_thaw_strict_false_dict. Retrieved 4/10 statements.
# Partially parsed test_thaw_strict_false_tuple. Retrieved 4/10 statements.
# Partially parsed test_thaw_nested_list_strict_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = {var_3: var_1}
    var_5 = [var_0, var_4]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_1.thaw(var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_0, var_1]
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.thaw(var_3)
    var_5 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = False
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = False
    var_3 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = False

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    var_2 = 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.thaw(var_0)
    var_2 = 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.thaw(var_3, var_4)
    var_6 = [var_4, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.thaw(var_4, var_5)
    var_7 = {var_0: var_5, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = [var_2, var_1]
    var_4 = [var_3]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = module_0.m()
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_1.thaw(var_3, var_4)
    var_6 = 'b'
    var_7 = {var_6: var_4}
    var_8 = {var_0: var_7}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list_empty. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_nested. Retrieved 11/14 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_1.pmap(var_8)
    var_10 = [var_3, var_4]

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = [var_1, var_2]

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple_with_elements. Retrieved 12/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = {var_1: var_2}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_4, var_5]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = ()

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = {var_4: var_5}
    var_11 = module_1.pmap(var_10)

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'

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
    var_8 = var_5

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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = [var_1, var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_mixed_args. Retrieved 12/34 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_non_container_args. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_nested_freezing. Retrieved 10/25 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'old'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = 10
    var_9 = 20
    var_10 = (var_8, var_9)
    var_11 = 0

def test_case_0():
    var_0 = 5
    var_1 = 10

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = (var_4, var_7)
    var_9 = {var_0: var_3, var_1: var_8}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/22 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 10/15 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
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

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = module_0.freeze(var_11)
    var_13 = {var_3, var_4}
    var_14 = module_1.pset(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_1: var_2}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_0, var_9]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_5]
    var_7 = True
    var_8 = module_1.freeze(var_6, var_7)
    var_9 = [var_2]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

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



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_strict_pmap. Retrieved 9/13 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_with_positional_arguments. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 5/6 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 8/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 12/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 9/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 2/10 statements.
# Failed to parse test_mutant_preserves_function_metadata.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = module_0.mutant(var_0)
    var_2 = 3
    var_3 = 4

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda lst: lst.append(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5, var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = lambda d: d.update(var_2)
    var_4 = module_0.mutant(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_5: var_7, var_6: var_8, var_0: var_1}
    var_11 = module_1.pmap(var_10)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = lambda s: s.add(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = [var_3, var_4, var_5, var_0]
    var_8 = module_1.pset(var_7)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = (var_0,)
    var_2 = lambda t: t + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = (var_4, var_5, var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 4
    var_2 = lambda obj: obj[var_0].append(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_5, var_6, var_1]

def test_case_0():
    var_0 = {}
    var_1 = 'key'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 1/8 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 2/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 2/7 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 2/8 statements.
# Partially parsed test_mutant_with_positional_and_keyword_arguments. Retrieved 4/17 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 1/9 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = set()
    var_1 = set()

def test_case_0():
    var_0 = 0
    var_1 = (var_0,)

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 3
    var_3 = 0

def test_case_0():
    var_0 = 'a'



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 5/6 statements.
# Partially parsed test_mutant_freezes_list_arg. Retrieved 8/10 statements.
# Partially parsed test_mutant_freezes_dict_arg. Retrieved 10/11 statements.
# Partially parsed test_mutant_freezes_set_arg. Retrieved 8/9 statements.
# Partially parsed test_mutant_freezes_tuple_arg. Retrieved 9/12 statements.
# Partially parsed test_mutant_freezes_nested_args. Retrieved 16/20 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 7/9 statements.
# Partially parsed test_mutant_freezes_kwargs_values. Retrieved 11/12 statements.
# Failed to parse test_mutant_preserves_function_name.
# Partially parsed test_mutant_with_no_args. Retrieved 3/4 statements.
# Partially parsed test_mutant_with_empty_args. Retrieved 6/7 statements.
# Partially parsed test_mutant_with_strict_false_implicitly. Retrieved 9/13 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = module_0.mutant(var_0)
    var_2 = 3
    var_3 = 4

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y - z
    var_1 = module_0.mutant(var_0)
    var_2 = 5
    var_3 = 3
    var_4 = 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = lambda lst: lst.append(var_4) or lst
    var_6 = module_0.mutant(var_5)
    var_7 = [var_0, var_1, var_2, var_4]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = lambda d: d.update(var_5) or d
    var_7 = module_0.mutant(var_6)
    var_8 = {var_0: var_1, var_3: var_4}
    var_9 = module_1.pmap(var_8)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = lambda s: s.add(var_3) or s
    var_5 = module_0.mutant(var_4)
    var_6 = [var_0, var_1, var_3]
    var_7 = module_1.pset(var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = lambda t: t[var_0].append(var_3) or t
    var_5 = module_0.mutant(var_4)
    var_6 = 10
    var_7 = (var_6, var_2)
    var_8 = [var_0, var_1, var_3]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = lambda obj: obj[var_0].append(var_9) or obj[var_1].add(var_10) or obj
    var_12 = module_0.mutant(var_11)
    var_13 = [var_2, var_3, var_9]
    var_14 = [var_5, var_6, var_10]
    var_15 = module_1.pset(var_14)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = lambda **kw: kw.update(var_2) or kw
    var_4 = module_0.mutant(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_5, var_8: var_6, var_0: var_1}
    var_10 = module_1.pmap(var_9)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0
    var_2 = module_0.mutant(var_1)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = lambda *args, **kwargs: (args, kwargs)
    var_1 = module_0.mutant(var_0)
    var_2 = ()
    var_3 = {}
    var_4 = module_1.pmap(var_3)
    var_5 = (var_2, var_4)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = lambda lst: lst[var_0].append(var_5) or lst
    var_7 = module_0.mutant(var_6)
    var_8 = [var_1, var_2, var_5]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 7/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 100
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = 2
    var_6 = 999



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/22 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 1/4 statements.
# Partially parsed test_mutant_decorator_with_defaultdict. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}

def test_case_0():
    pass

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5
    assert var_0 == 5

def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]
    var_5 = {var_0: var_4}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 11/18 statements.
# Partially parsed test_mutant_return_frozen. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_nested_mutables. Retrieved 12/20 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'initial'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = 'factor'
    var_9 = {var_3: var_4, var_8: var_6}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_3, var_2]
    var_10 = {var_5, var_6}
    var_11 = module_0.pset(var_10)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_freeze_pmap_strict. Retrieved 9/13 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 11/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 5/8 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = module_0.freeze(var_5)
    var_7 = {var_1: var_2}
    var_8 = module_1.pmap(var_7)
    var_9 = [var_0, var_8, var_4]

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_2, var_3]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_1: var_2}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_0, var_9]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = {var_1: var_2}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_6, var_9]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'

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
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = {var_0: var_4}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 12/17 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 7/10 statements.
# Partially parsed test_mutant_preserves_original_input. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_set. Retrieved 10/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 8/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/16 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 8/12 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: [x]
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x=[]: x.append(var_0) or x
    var_2 = module_0.mutant(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_3, var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = lambda x, y: {var_0: x, var_1: y}
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = [var_4]
    var_10 = {var_6: var_7}
    var_11 = module_1.pmap(var_10)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda a, b=[]: a + b
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = [var_2, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = lambda d: d[var_0].append(var_5) or d
    var_7 = module_0.mutant(var_6)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = {var_0}
    var_2 = lambda s: s.union(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_4, var_5, var_6}
    var_8 = {var_4, var_5, var_6, var_0}
    var_9 = module_1.pset(var_8)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = (var_0,)
    var_2 = lambda t: t + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = (var_4, var_5, var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 3
    var_2 = lambda d: d[var_0].append(var_1) or d
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_5, var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = lambda x: x.append(var_3) or x
    var_5 = False
    var_6 = module_0.mutant(var_4)
    var_7 = [var_0, var_1, var_3]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 5/6 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 8/10 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 12/13 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 9/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/17 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 7/10 statements.
# Failed to parse test_mutant_preserves_function_name.
# Partially parsed test_mutant_with_empty_args. Retrieved 3/4 statements.
# Partially parsed test_mutant_with_no_return. Retrieved 4/5 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 8/9 statements.
# Partially parsed test_mutant_freezes_kwargs_dict_values. Retrieved 7/11 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = module_0.mutant(var_0)
    var_2 = 3
    var_3 = 4

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda lst: lst.append(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5, var_0]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = lambda d: d.update(var_2)
    var_4 = module_0.mutant(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_5: var_7, var_6: var_8, var_0: var_1}
    var_11 = module_1.pmap(var_10)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = lambda s: s.add(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = [var_3, var_4, var_5, var_0]
    var_8 = module_1.pset(var_7)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = lambda x: x[var_0].append(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'a'
    var_8 = {var_7: var_4}
    var_9 = [var_6, var_8]
    var_10 = [var_4, var_5, var_1]
    var_11 = {var_7: var_4}
    var_12 = module_1.pmap(var_11)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0
    var_2 = module_0.mutant(var_1)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = module_0.mutant(var_1)
    var_3 = 5

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = (var_0,)
    var_2 = lambda t: t + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = (var_4, var_5, var_6)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = lambda **kwargs: kwargs[var_0].append(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 0
    var_5 = [var_4]
    var_6 = [var_4, var_1]



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_does_not_mutate_inputs. Retrieved 20/30 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_1.pmap(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pset(var_9)
    var_11 = {var_5: var_0, var_6: var_1}
    var_12 = module_1.pmap(var_11)
    var_13 = 4
    var_14 = [var_0, var_1, var_2, var_13]
    var_15 = module_0.pset(var_14)
    var_16 = 'new'
    var_17 = 100
    var_18 = {var_5: var_0, var_6: var_1, var_16: var_17}
    var_19 = module_1.pmap(var_18)



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_decorator_does_not_mutate_inputs. Retrieved 19/29 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.pset(var_8)
    var_10 = {var_5: var_0}
    var_11 = module_1.pmap(var_10)
    var_12 = 4
    var_13 = [var_0, var_1, var_2, var_12]
    var_14 = module_0.pset(var_13)
    var_15 = 'new'
    var_16 = 99
    var_17 = {var_5: var_0, var_15: var_16}
    var_18 = module_1.pmap(var_17)



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_with_positional_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 11/18 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 6/11 statements.
# Failed to parse test_mutant_preserves_function_metadata.
# Partially parsed test_mutant_with_no_arguments. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_mutables. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_kwargs_values. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'initial'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = 'factor'
    var_9 = {var_3: var_4, var_8: var_6}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_3, var_1: var_2}
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
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = 'y'
    var_7 = [var_3, var_4]



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/25 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/10 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 10/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = {var_4, var_5}
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = {var_4, var_5}
    var_9 = module_0.pset(var_8)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_with_defaultdict_and_strict_true. Retrieved 7/12 statements.
# Partially parsed test_freeze_with_defaultdict_and_strict_false. Retrieved 6/10 statements.
# Partially parsed test_freeze_with_pmap_and_strict_true. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = [var_5, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False

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



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------






# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 7/13 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_multiple_args. Retrieved 12/23 statements.
# Partially parsed test_mutant_decorator_nested_structures. Retrieved 12/24 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 6/13 statements.
# Partially parsed test_mutant_decorator_no_side_effects_on_frozen_inputs. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_1, var_2]

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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = 0
    var_7 = 99
    var_8 = [var_0, var_1, var_7]
    var_9 = 100
    var_10 = {var_3, var_4, var_9}
    var_11 = module_0.pset(var_10)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = {var_4}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'changed'
    var_8 = [var_7]
    var_9 = 999
    var_10 = {var_4, var_9}
    var_11 = module_0.pset(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)
    var_6 = {var_0: var_1}
    var_7 = module_1.pmap(var_6)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_freeze_defaultdict_strict_true. Retrieved 7/12 statements.
# Partially parsed test_freeze_defaultdict_strict_false. Retrieved 6/10 statements.
# Partially parsed test_freeze_pmap_strict_true. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = [var_5, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False

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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 13/21 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_returns_frozen_result. Retrieved 1/8 statements.
# Partially parsed test_mutant_decorator_handles_no_arguments. Retrieved 1/8 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 11/22 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old'
    var_6 = {var_4: var_5}
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_5, var_9: var_10}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'set'

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = [var_5]
    var_7 = (var_4, var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'changed'
    var_10 = [var_9]



# Parsed testcases at query #38
#--------------------------






