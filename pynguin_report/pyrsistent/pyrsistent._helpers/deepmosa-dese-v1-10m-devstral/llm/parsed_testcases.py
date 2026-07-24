####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 11/16 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 7/12 statements.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = {var_3: var_4}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_1, var_2, var_9]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_2, var_3]
    var_10 = {var_5: var_6}
    var_11 = module_0.pmap(var_10)

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
    var_0 = []
    var_1 = {}
    var_2 = module_0.pmap(var_1)
    var_3 = set()
    var_4 = module_1.pset(var_3)
    var_5 = ()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 13/14 statements.
# Partially parsed test_freeze_pvector. Retrieved 4/6 statements.
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
    var_2 = 2
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
    var_2 = []
    var_3 = module_1.pset(var_2)

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
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = {var_1: var_2}
    var_10 = module_1.pmap(var_9)
    var_11 = (var_4, var_5)
    var_12 = [var_0, var_10, var_11]

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_mutant_with_empty_args_and_kwargs.
# Partially parsed test_mutant_with_list_arg. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 23/31 statements.
# Partially parsed test_mutant_with_pvector_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_pmap_arg. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_pset_arg. Retrieved 5/8 statements.
# Partially parsed test_mutant_with_non_container_arg. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_2, var_3]
    var_10 = [var_5, var_6]
    var_11 = module_0.pset(var_10)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

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
    var_9 = 6
    var_10 = 7
    var_11 = [var_10]
    var_12 = (var_9, var_11)
    var_13 = 'arg1'
    var_14 = 'arg2'
    var_15 = 'kwarg1'
    var_16 = 'kwarg2'
    var_17 = [var_0, var_1]
    var_18 = {var_3: var_4}
    var_19 = module_0.pmap(var_18)
    var_20 = [var_6, var_7]
    var_21 = module_1.pset(var_20)
    var_22 = [var_10]

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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #6
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

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 10/17 statements.


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
    var_0 = []
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_value'
    var_1 = 'existing_key'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = [var_0, var_5]
    var_8 = {var_2: var_3, var_5: var_6}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 13/19 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 15/21 statements.
# Partially parsed test_mutant_nested_structures. Retrieved 18/26 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pvector. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_pmap. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_pset. Retrieved 7/10 statements.
# Partially parsed test_mutant_no_mutation. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/12 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0]
    var_4 = [var_1]
    var_5 = [var_2]
    var_6 = [var_0, var_1, var_2]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = 'b'
    var_10 = {var_9: var_1}
    var_11 = 'c'
    var_12 = {var_11: var_2}
    var_13 = {var_7: var_0, var_9: var_1, var_11: var_2}
    var_14 = module_0.pmap(var_13)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 2
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = 4
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = (var_8, var_11)
    var_13 = {var_0: var_7, var_1: var_12}
    var_14 = {var_4: var_5}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_2, var_3, var_15]
    var_17 = [var_9, var_10]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset(var_3)
    var_5 = {var_0, var_1, var_2}
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]



# Parsed testcases at query #11
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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 13/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 4/6 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = {var_1: var_2}
    var_10 = module_1.pmap(var_9)
    var_11 = (var_4, var_5)
    var_12 = [var_0, var_10, var_11]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = [var_3, var_1]

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



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_positional_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mutable_kwargs. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_nested_mutable_args. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_pvector_arg. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_arg. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_pset_arg. Retrieved 8/11 statements.
# Failed to parse test_mutant_with_strict_false.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 3
    var_1 = 4

def test_case_0():
    var_0 = 3
    var_1 = 4

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

def test_case_0():
    var_0 = 'lst'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_0: var_2, var_1: var_3, var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #18
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_freeze_predicate_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = True



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/20 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_persistent_types. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/12 statements.
# Failed to parse test_mutant_preserves_function_name_and_doc.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0]
    var_5 = [var_1]
    var_6 = [var_0]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '-'
    var_3 = [var_0]
    var_4 = [var_1]
    var_5 = [var_0]
    var_6 = [var_1]
    var_7 = [var_0]
    var_8 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = module_0.pmap(var_4)
    var_6 = 'new'
    var_7 = {var_3: var_0, var_6: var_0}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 3
    var_1 = 4
    var_2 = {var_0, var_1}
    var_3 = 1
    var_4 = 2
    var_5 = {var_3, var_4, var_0, var_1}
    var_6 = module_0.pset(var_5)



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 15/19 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/11 statements.


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
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'list'
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
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_7: var_2, var_8: var_3, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_4: var_0, var_5: var_1, var_6: var_12}
    var_14 = module_0.pmap(var_13)

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
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_values. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 12/15 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/8 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

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
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = [var_2, var_3]
    var_11 = (var_5, var_6)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_pvector. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_pmap. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = (var_1, var_2, var_1)
    var_6 = {var_0: var_5}
    var_7 = module_0.pmap(var_6)

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
    var_4 = [var_0, var_1, var_2, var_0]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 1
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 11/17 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict. Retrieved 13/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/9 statements.


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
    var_2 = 2
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
    var_2 = []
    var_3 = module_1.pset(var_2)

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

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

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
    var_11 = module_1.pmap(var_10)
    var_12 = [var_0, var_11]

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
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_immutable_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_dict_args. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/11 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 4
    var_7 = 'arg1'
    var_8 = 'arg2'
    var_9 = 'kwargs'
    var_10 = [var_0, var_1]
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)
    var_13 = 'extra'
    var_14 = {var_13: var_6}
    var_15 = module_0.pmap(var_14)

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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_false. Retrieved 12/15 statements.


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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

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
    var_7 = {var_6: var_0}
    var_8 = module_0.pmap(var_7)
    var_9 = module_1.freeze(var_8, var_4)
    var_10 = {var_6: var_0}
    var_11 = module_0.pmap(var_10)



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mutable_keyword_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 16/24 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 'b'
    var_5 = {var_0: var_1, var_4: var_3}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2, var_1]

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
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = {var_5: var_0, var_6: var_1, var_7: var_2}
    var_9 = 4
    var_10 = [var_0, var_1, var_2, var_9]
    var_11 = module_0.pset(var_10)
    var_12 = [var_0, var_1, var_2, var_9]
    var_13 = 'd'
    var_14 = {var_5: var_0, var_6: var_1, var_7: var_2, var_13: var_9}
    var_15 = module_1.pmap(var_14)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 16/28 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 15/24 statements.


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
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = [var_2, var_3]
    var_8 = 'b'
    var_9 = [var_0, var_2]
    var_10 = 'c'
    var_11 = {var_10: var_3}
    var_12 = {var_1: var_9, var_8: var_11}
    var_13 = [var_0, var_2]
    var_14 = {var_10: var_3}
    var_15 = module_0.pmap(var_14)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

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
    var_12 = 'c'
    var_13 = 'd'
    var_14 = [var_5, var_6]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 14/19 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_key'
    var_6 = 42
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 11/17 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/9 statements.


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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_1, var_2]
    var_9 = {var_0: var_8}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 8/9 statements.


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



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 22/28 statements.
# Partially parsed test_mutant_with_non_container_types. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 1/6 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 11/16 statements.


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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = set(var_8)
    var_10 = {var_1: var_5, var_2: var_9}
    var_11 = [var_0, var_10]
    var_12 = [var_3, var_4]
    var_13 = {var_6, var_7}
    var_14 = module_0.pset(var_13)

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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'y'
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = set(var_9)
    var_11 = {var_6: var_10}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'kwargs'
    var_15 = [var_0, var_1]
    var_16 = {var_3: var_4}
    var_17 = module_0.pmap(var_16)
    var_18 = {var_7, var_8}
    var_19 = module_1.pset(var_18)
    var_20 = {var_6: var_19}
    var_21 = module_0.pmap(var_20)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.


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
    var_0 = 5
    var_1 = 15

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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 8/9 statements.


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



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_mutant_decorator_returns_callable.


def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 21/27 statements.


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
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'x'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)

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
    var_8 = [var_6, var_7]
    var_9 = set(var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'kwargs'
    var_13 = [var_0, var_1]
    var_14 = {var_3: var_4}
    var_15 = module_0.pmap(var_14)
    var_16 = 'c'
    var_17 = [var_6, var_7]
    var_18 = module_1.pset(var_17)
    var_19 = {var_16: var_18}
    var_20 = module_0.pmap(var_19)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_decorator_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_decorator_with_list_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_decorator_with_dict_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_decorator_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 8/15 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

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
    var_0 = 'values'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = [var_1, var_4, var_5]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector_with_elements. Retrieved 5/8 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/9 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = {var_1: var_2}
    var_7 = [var_0, var_6]

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

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_list_arg. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 21/31 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 19/31 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 3/7 statements.


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
    var_5 = 5
    var_6 = {var_0, var_1, var_2, var_4, var_5}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'inner_list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = {var_4, var_5}
    var_7 = 6
    var_8 = 7
    var_9 = (var_7, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'kwargs'
    var_13 = [var_1, var_2]
    var_14 = 'c'
    var_15 = 'd'
    var_16 = {var_4, var_5}
    var_17 = module_0.pset(var_16)
    var_18 = (var_7, var_8)
    var_19 = {var_14: var_17, var_15: var_18}
    var_20 = module_1.pmap(var_19)

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
    var_9 = 6
    var_10 = 7
    var_11 = (var_9, var_10)
    var_12 = [var_0, var_1]
    var_13 = {var_3: var_4}
    var_14 = module_0.pmap(var_13)
    var_15 = {var_6, var_7}
    var_16 = module_1.pset(var_15)
    var_17 = (var_9, var_10)
    var_18 = 0

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 11/17 statements.
# Partially parsed test_mutant_nested. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_list_operations. Retrieved 6/11 statements.
# Partially parsed test_mutant_set_operations. Retrieved 6/10 statements.
# Partially parsed test_mutant_tuple_operations. Retrieved 4/7 statements.
# Partially parsed test_mutant_mixed_types. Retrieved 19/28 statements.


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

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'nested'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_3, var_4, var_5}
    var_8 = 'value'
    var_9 = 5
    var_10 = {var_8: var_9}
    var_11 = {var_0: var_6, var_1: var_7, var_2: var_10}
    var_12 = 4
    var_13 = [var_3, var_4, var_5, var_12]
    var_14 = [var_3, var_4, var_5, var_12]
    var_15 = module_0.pset(var_14)
    var_16 = 6
    var_17 = {var_8: var_16}
    var_18 = module_1.pmap(var_17)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/13 statements.


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

# Partially parsed test_mutant_with_simple_types. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/27 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'

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
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_2, var_3, var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_with_dict. Retrieved 8/9 statements.


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



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_mutant_with_empty_args_and_kwargs.
# Partially parsed test_mutant_with_simple_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_simple_kwargs. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_mutable_return. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_mutable_return_nested. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #11
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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.


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
    var_2 = 2
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/7 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_mixed_types. Retrieved 21/26 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.pmap(var_2)

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

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = [var_2, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = 7
    var_12 = [var_10, var_11]
    var_13 = (var_9, var_12)
    var_14 = {var_0: var_8, var_1: var_13}
    var_15 = [var_4, var_5]
    var_16 = module_0.pset(var_15)
    var_17 = (var_3, var_16)
    var_18 = [var_2, var_17]
    var_19 = [var_10, var_11]
    var_20 = module_1.freeze(var_14)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_set_arguments. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuple_arguments. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 10/16 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 'values'
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
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.pmap(var_8)

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
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_5: var_0}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/7 statements.
# Failed to parse test_mutant_returns_frozen.
# Partially parsed test_mutant_with_pvector_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_pset_input. Retrieved 8/11 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 4/8 statements.


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
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 'value'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

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
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_mutant_decorator_returns_callable.


def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_mutant_predicate_at_line_1.




# Parsed testcases at query #26
#--------------------------

# Failed to parse test_mutant_with_empty_function.
# Partially parsed test_mutant_with_simple_arguments. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 4/8 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 4
    var_7 = 'arg1'
    var_8 = 'arg2'
    var_9 = 'kwargs'
    var_10 = [var_0, var_1]
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)
    var_13 = 'extra'
    var_14 = {var_13: var_6}
    var_15 = module_0.pmap(var_14)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 11/17 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_pvector_with_strict_false. Retrieved 5/7 statements.
# Partially parsed test_freeze_pvector_with_strict_true. Retrieved 7/12 statements.


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
    var_3 = [var_0, var_1, var_2]
    var_4 = False

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = [var_1, var_2]

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = module_1.freeze(var_4)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_with_list_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_pvector_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_pmap_arg. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_pset_arg. Retrieved 4/7 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}

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
    var_4 = {var_3: var_0}
    var_5 = {var_0, var_1}
    var_6 = (var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'lst'
    var_1 = 'd'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_4, var_1: var_6}

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
    var_2 = {var_0, var_1}
    var_3 = module_0.pset(var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_basic_operation. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 7/13 statements.


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
    var_0 = 'values'
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
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = [var_1, var_2]
    var_6 = [var_0, var_5]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_mutable_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/27 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 10/16 statements.


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
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = 'd'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_2]
    var_8 = {var_4: var_5}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_freeze_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_with_non_empty_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_with_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 9/10 statements.
# Partially parsed test_freeze_with_defaultdict. Retrieved 7/10 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_func.
# Failed to parse test_mutant_decorator_preserves_function_signature.




# Parsed testcases at query #39
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_with_list_arg. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/7 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}

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
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.


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
    var_0 = 'inner'
    var_1 = 'other'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'value'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 'changed'
    var_9 = [var_8, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/16 statements.
# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_set_argument. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.


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
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

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
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_9}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_non_empty_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_with_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_nested_dict_with_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 4/6 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_defaultdict. Retrieved 6/12 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 18/23 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]

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
    var_2 = [var_0, var_1]
    var_3 = False

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pset(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)

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
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = (var_7, var_10)
    var_12 = {var_0: var_6, var_1: var_11}
    var_13 = {var_3: var_4}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8, var_9]
    var_17 = module_1.freeze(var_12)



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 11/17 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_pvector_with_strict_false. Retrieved 5/7 statements.
# Partially parsed test_freeze_pvector_with_strict_true. Retrieved 7/12 statements.


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
    var_3 = [var_0, var_1, var_2]
    var_4 = False

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
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 3/6 statements.
# Failed to parse test_mutant_with_no_args.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = (var_7,)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/12 statements.


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
    var_0 = 'existing_value'
    var_1 = 'existing_key'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 'value'
    var_5 = 'combined'
    var_6 = 'extra'
    var_7 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 42
    var_1 = 'string'
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 15/22 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_set. Retrieved 9/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 9/15 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 'set'
    var_5 = [var_0, var_1, var_2]
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 12/20 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.


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
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = [var_2, var_3, var_2]
    var_9 = 'new_key'
    var_10 = 'value'
    var_11 = {var_5: var_2, var_9: var_10}
    var_12 = module_0.pmap(var_11)

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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 20
    var_6 = 'arg1'
    var_7 = 'arg2'
    var_8 = [var_0, var_1, var_5]
    var_9 = 'new'
    var_10 = {var_3: var_0, var_9: var_5}
    var_11 = module_0.pmap(var_10)

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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 3/7 statements.
# Failed to parse test_mutant_with_mutable_return.
# Partially parsed test_mutant_with_pvector_input. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 4/9 statements.


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
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #15
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = callable(var_1)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 9/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_pvector. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_pset. Retrieved 8/11 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'lst'
    var_1 = 'd'
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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4

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
    var_6 = 'c'
    var_7 = 3
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



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_basic_functionality. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/20 statements.
# Partially parsed test_mutant_with_tuple_and_set. Retrieved 10/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/20 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_pvector_and_pmap. Retrieved 11/19 statements.


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
    var_0 = 'values'
    var_1 = 'metadata'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'count'
    var_7 = {var_6: var_4}
    var_8 = {var_0: var_5, var_1: var_7}
    var_9 = 4
    var_10 = [var_2, var_3, var_4, var_9]
    var_11 = {var_6: var_4}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_1]
    var_8 = {var_4, var_5}
    var_9 = module_0.pset(var_8)

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
    var_12 = True

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_2, var_3, var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.pmap(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_0, var_1, var_0]
    var_8 = 'new'
    var_9 = {var_3: var_4, var_8: var_1}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 13/19 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 9/12 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 10/13 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = {var_2: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_6, var_2, var_3]

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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap(var_7)

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
    var_4 = 'new'
    var_5 = 100
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.


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
    var_0 = 'key'
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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = {var_2: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 6
    var_5 = 9
    var_6 = [var_2, var_4, var_5]

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_with_simple_list. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_nested_structure. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_defaultdict. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_pvector_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_pmap_input. Retrieved 8/11 statements.


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
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 42
    var_5 = {var_2: var_4, var_3: var_1}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]

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
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 42
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

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
    var_5 = 42
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_and_set. Retrieved 10/14 statements.


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

import pyrsistent._transformations as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_1, var_3, var_3]
    var_5 = (var_2, var_4)
    var_6 = module_0.transform(var_5)
    var_7 = [var_0, var_1]
    var_8 = [var_1, var_3]
    var_9 = module_1.pset(var_8)



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/16 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/17 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/10 statements.


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
    var_0 = 42
    var_1 = 'hello'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_2, var_3, var_4)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = {var_1, var_3}
    var_5 = [var_0, var_1, var_3]
    var_6 = module_0.pset(var_5)



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_positional_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_mutable_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_mutable_structures. Retrieved 15/27 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_returns_immutable_version. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 3/9 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.pmap(var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 12/18 statements.
# Partially parsed test_mutant_decorator_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 7/14 statements.
# Partially parsed test_mutant_decorator_with_mixed_types. Retrieved 14/17 statements.
# Partially parsed test_mutant_decorator_with_strict_false. Retrieved 6/12 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_1, var_2]

import pyrsistent._transformations as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.transform(var_3)
    var_5 = 4
    var_6 = 6
    var_7 = [var_1, var_5, var_6]
    var_8 = (var_0, var_1, var_2)
    var_9 = module_0.transform(var_8)
    var_10 = {var_0, var_1, var_2}
    var_11 = module_0.transform(var_10)
    var_12 = {var_1, var_5, var_6}
    var_13 = module_1.pset(var_12)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 7/11 statements.
# Partially parsed test_mutant_nested. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/17 statements.
# Partially parsed test_mutant_empty_args. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_0]
    var_7 = [var_2]
    var_8 = [var_4]

def test_case_0():
    var_0 = []

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #37
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



# Parsed testcases at query #38
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



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_decorator_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 9/13 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_decorator_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 17/25 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_decorator_with_pvector_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_with_pmap_argument. Retrieved 8/11 statements.
# Partially parsed test_mutant_decorator_with_pset_argument. Retrieved 8/11 statements.


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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = module_0.pset(var_6)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_mixed_nested_structures. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 5/7 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_pmap_with_strict_true. Retrieved 9/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

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
    var_4 = False

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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 7/11 statements.
# Partially parsed test_mutant_nested. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_empty_args. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.


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
    var_0 = []

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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/7 statements.
# Failed to parse test_mutant_with_non_strict_freeze.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_decorator_preserves_false_predicate. Retrieved 1/1 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = False



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector_with_elements. Retrieved 5/8 statements.


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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_mutant_with_list_arg. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 15/23 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_value'
    var_1 = 'existing_key'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 'value'
    var_6 = [var_0, var_1, var_0]
    var_7 = 'key'
    var_8 = {var_3: var_0, var_7: var_5}
    var_9 = module_0.pmap(var_8)
    var_10 = 'existing_kwarg'
    var_11 = 'kwarg'
    var_12 = 'kwarg_value'
    var_13 = {var_10: var_5, var_11: var_12}
    var_14 = module_0.pmap(var_13)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 15/27 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 10/23 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/14 statements.


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

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_value'
    var_1 = 'existing_key'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 'test'
    var_6 = 'arg1'
    var_7 = 'arg2'
    var_8 = 'kwarg1'
    var_9 = 'modified'
    var_10 = [var_0, var_1, var_9]
    var_11 = 'new_key'
    var_12 = 'new_value'
    var_13 = {var_3: var_0, var_11: var_12}
    var_14 = module_0.pmap(var_13)

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
    var_2 = module_0.pmap(var_1)
    var_3 = []
    var_4 = module_1.pset(var_3)
    var_5 = ()
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_decorator_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_decorator_with_mixed_args_and_kwargs. Retrieved 9/13 statements.
# Partially parsed test_mutant_decorator_with_no_args. Retrieved 5/9 statements.


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
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_0: var_1, var_5: var_3, var_6: var_4}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.pmap(var_1)
    var_3 = []
    var_4 = module_1.pset(var_3)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_thaw_pvector_to_list. Retrieved 3/6 statements.
# Partially parsed test_thaw_pset_to_set. Retrieved 3/6 statements.
# Partially parsed test_thaw_nested_pvector. Retrieved 3/6 statements.
# Partially parsed test_thaw_nested_pmap. Retrieved 2/6 statements.
# Partially parsed test_thaw_nested_tuple. Retrieved 3/7 statements.
# Partially parsed test_thaw_non_strict_pvector. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_1.thaw(var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.thaw(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'hello'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.thaw(var_3, var_4)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.thaw(var_4, var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = False

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = False
    var_3 = module_1.thaw(var_1, var_2)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.thaw(var_2, var_3)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.thaw(var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 11/17 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 6/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/9 statements.


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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = False

import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pset(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_containers. Retrieved 10/16 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 6/9 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_non_strict_nested_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/10 statements.


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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

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
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = {var_1: var_5}
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_3]

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = [var_0, var_1, var_2]

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_1, var_2]
    var_8 = [var_0, var_7]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = {var_1: var_2}
    var_8 = {var_0: var_7}
    var_9 = module_1.pmap(var_8)

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
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = 3.14
    var_5 = module_0.freeze(var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 8/12 statements.
# Partially parsed test_mutant_nested. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/13 statements.
# Partially parsed test_mutant_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_set_handling. Retrieved 3/6 statements.
# Partially parsed test_mutant_tuple_preservation. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'b'
    var_7 = {var_6: var_2}

def test_case_0():
    var_0 = 'values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = 7
    var_9 = [var_7, var_8]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 13/16 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 8/14 statements.


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
    var_0 = 'inner'
    var_1 = 'other'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'value'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'changed'
    var_8 = [var_7, var_3]

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
    var_4 = [var_0, var_1, var_2, var_0]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = {}
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = set()
    var_6 = []
    var_7 = module_1.pset(var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 11/20 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0]
    var_5 = [var_1]
    var_6 = [var_0]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'result'
    var_7 = [var_1, var_2, var_3]

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
    var_8 = [var_0, var_1]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)

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
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)



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

# Partially parsed test_mutant_with_list_arg. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_arg. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 16/30 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_already_frozen. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 17/35 statements.


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

def test_case_0():
    var_0 = 'list'
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
    var_2 = 'existing_value'
    var_3 = 'existing_key'
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = module_0.pmap(var_6)

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_0, var_1}
    var_6 = (var_0, var_1)
    var_7 = 4
    var_8 = [var_0, var_1, var_7]
    var_9 = 'new_key'
    var_10 = {var_3: var_0, var_9: var_7}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_0, var_1, var_7}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1, var_7)
    var_15 = 0
    var_16 = 3



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 10/16 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/7 statements.


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
    var_2 = 2
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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)

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
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = {var_1: var_5}
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_3]

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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/8 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pmap()

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
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_mutable_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_set_arguments. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple_arguments. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 16/22 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 7/13 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'extra'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_0}
    var_6 = module_0.pmap(var_5)

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
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_0]
    var_12 = {var_2: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = [var_5, var_6]
    var_15 = module_1.pset(var_14)

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
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = [var_0, var_1, var_2]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_basic_functionality. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_sets. Retrieved 7/10 statements.
# Partially parsed test_mutant_with_tuples. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = [var_4, var_5]
    var_7 = [var_4, var_5]

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



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_mutant_with_no_args.
# Partially parsed test_mutant_with_positional_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_mutable_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_nested_mutable_args. Retrieved 8/16 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_set_arg. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple_arg. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_2, var_3]

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
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_decorator_with_list_arg. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_dict_arg. Retrieved 9/13 statements.
# Partially parsed test_mutant_decorator_with_set_arg. Retrieved 7/11 statements.
# Partially parsed test_mutant_decorator_with_tuple_arg. Retrieved 4/7 statements.
# Partially parsed test_mutant_decorator_with_mixed_args. Retrieved 17/25 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 8/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 11/16 statements.
# Partially parsed test_mutant_decorator_preserves_immutable_types. Retrieved 3/8 statements.


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
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = {var_0, var_1, var_2}
    var_8 = (var_0, var_1, var_2)
    var_9 = 4
    var_10 = [var_0, var_1, var_2, var_9]
    var_11 = 'c'
    var_12 = {var_4: var_0, var_5: var_1, var_11: var_2}
    var_13 = module_0.pmap(var_12)
    var_14 = [var_0, var_1, var_2, var_9]
    var_15 = module_1.pset(var_14)
    var_16 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = [var_0, var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_1: var_0, var_2: var_3, var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_0, var_9]

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = True



# Parsed testcases at query #19
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = callable(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fn. Retrieved 1/3 statements.
# Partially parsed test_mutant_decorator_preserves_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = 5

def test_case_0():
    var_0 = 1
    var_1 = 5



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_func. Retrieved 1/3 statements.
# Partially parsed test_mutant_decorator_preserves_original_function. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 15/22 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 16/22 statements.


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
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_0]
    var_12 = {var_2: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_5, var_6}
    var_15 = module_1.pset(var_14)



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 6/11 statements.


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
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_1, var_2, var_3, var_6]

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #32
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 14/19 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.


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

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 18/22 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 10/16 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 10/16 statements.


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
    var_10 = [var_1, var_4]
    var_11 = 4
    var_12 = [var_6, var_11]
    var_13 = 5
    var_14 = 'extra'
    var_15 = 0
    var_16 = {var_14: var_13, var_15: var_1, var_1: var_4, var_4: var_6, var_6: var_11}
    var_17 = module_0.pmap(var_16)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

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
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_1, var_2]
    var_8 = {var_4: var_5}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_5: var_0}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 15/23 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 3/8 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4

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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_mixed_structures. Retrieved 11/17 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 6/9 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 6/8 statements.
# Partially parsed test_freeze_non_strict_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_non_strict_pmap. Retrieved 5/8 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/7 statements.


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
    var_4 = True
    var_5 = [var_4, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {var_0: var_1}

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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args. Retrieved 15/25 statements.
# Partially parsed test_mutant_returns_non_frozen_type. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/9 statements.


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
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'list'
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
    var_1 = 'a'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)

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
    var_13 = [var_0, var_1, var_2, var_7]
    var_14 = module_1.pset(var_13)

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_with_list. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 11/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 16/23 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 13/21 statements.


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
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 10
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 4
    var_9 = [var_2, var_3, var_4, var_8]
    var_10 = 11

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = 'd'
    var_12 = [var_0]
    var_13 = {var_2: var_3}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 42
    var_1 = 'string'
    var_2 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new_value'
    var_11 = {var_4: var_10}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_mutant_predicate.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 15/22 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 11/18 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = 'b'
    var_9 = {var_4: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = module_0.pset(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_mutant_decorator_basic. Retrieved 13/19 statements.
# Partially parsed test_mutant_decorator_nested_structures. Retrieved 9/19 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 11/19 statements.
# Failed to parse test_mutant_decorator_empty_args.
# Partially parsed test_mutant_decorator_with_set. Retrieved 8/11 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/7 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 'key'
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
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 'd'
    var_10 = [var_3, var_4]

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_basic. Retrieved 2/5 statements.
# Partially parsed test_mutant_list. Retrieved 5/10 statements.
# Partially parsed test_mutant_dict. Retrieved 7/11 statements.
# Partially parsed test_mutant_nested. Retrieved 8/15 statements.
# Partially parsed test_mutant_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_no_mutation. Retrieved 1/4 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 42



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_mutant_with_list_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_dict_input. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_tuple_input. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_set_input. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_mixed_inputs. Retrieved 17/26 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/12 statements.


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

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = (var_0, var_1, var_2)
    var_7 = {var_0, var_1, var_2}
    var_8 = 4
    var_9 = [var_0, var_1, var_2, var_8]
    var_10 = 'new_key'
    var_11 = 'new_value'
    var_12 = {var_4: var_0, var_10: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = (var_0, var_1, var_2, var_8)
    var_15 = {var_0, var_1, var_2, var_8}
    var_16 = module_1.pset(var_15)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

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



# Parsed testcases at query #48
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_with_no_args. Retrieved 1/2 statements.
# Failed to parse test_func.
# Partially parsed test_mutant_with_positional_args. Retrieved 2/2 statements.
# Failed to parse test_func.
# Partially parsed test_mutant_with_keyword_args. Retrieved 2/2 statements.
# Failed to parse test_func.
# Partially parsed test_mutant_with_mixed_args. Retrieved 3/3 statements.
# Partially parsed test_func. Retrieved 1/3 statements.
# Partially parsed test_mutant_with_mutable_args. Retrieved 7/3 statements.
# Failed to parse test_func.
# Partially parsed test_mutant_with_dict_args. Retrieved 7/2 statements.
# Partially parsed test_func. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/6 statements.
# Partially parsed test_mutant_return_value_is_frozen. Retrieved 8/5 statements.
# Partially parsed test_mutant_with_strict_false. Retrieved 13/7 statements.


def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 3
    var_1 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 4
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = [var_1, var_2, var_3, var_5]

def test_case_0():
    var_0 = 4
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = [var_1, var_2, var_3, var_5]

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
    var_0 = 'list'
    var_1 = 4
    var_2 = 'list'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'a'
    var_9 = {var_8: var_4}
    var_10 = {var_2: var_7, var_3: var_9}
    var_11 = 4
    var_12 = [var_4, var_5, var_6, var_11]
    var_13 = 'new_key'
    var_14 = 'new_value'
    var_15 = {var_8: var_4, var_13: var_14}
    var_16 = module_0.pmap(var_15)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 4
    var_2 = 'list'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'a'
    var_9 = {var_8: var_4}
    var_10 = {var_2: var_7, var_3: var_9}
    var_11 = 4
    var_12 = [var_4, var_5, var_6, var_11]
    var_13 = 'new_key'
    var_14 = 'new_value'
    var_15 = {var_8: var_4, var_13: var_14}
    var_16 = module_0.pmap(var_15)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'a'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = {var_6: var_10}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'a'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = {var_6: var_10}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_freeze_defaultdict_conversion. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_mutant_with_simple_function. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict_arguments. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 2/6 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



