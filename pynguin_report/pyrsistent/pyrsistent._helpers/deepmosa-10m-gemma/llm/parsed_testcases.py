####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 13/24 statements.
# Partially parsed test_mutant_protects_against_mutation_in_function_body. Retrieved 6/12 statements.
# Partially parsed test_mutant_returns_frozen_output. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]
    var_6 = 'key'
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = [var_2, var_3]
    var_12 = [var_7, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 99
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True

def test_case_0():
    var_0 = 10
    var_1 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_simple_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_and_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 12/17 statements.
# Partially parsed test_freeze_strict_true_defaultdict. Retrieved 5/11 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_1, var_2]
    var_11 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = [var_0]
    var_5 = {var_4}
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_3 == var_6)
    assert var_7 is True
    var_8 = 2
    var_9 = (var_0, var_8)
    var_10 = {var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = (var_0, var_8)
    var_13 = {var_12}
    var_14 = module_1.pset(var_13)
    var_15 = bool(var_11 == var_14)
    assert var_15 is True

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_dict_nested. Retrieved 13/16 statements.
# Partially parsed test_freeze_list_simple. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_nested. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 11/13 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 15/19 statements.


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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

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
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = {var_4: var_5}
    var_11 = module_1.pmap(var_10)

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = (var_0, var_2, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_1]
    var_9 = {var_3: var_4}
    var_10 = module_1.pmap(var_9)

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
    var_8 = bool(var_7 == 3.14)
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
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 4
    var_5 = (var_4,)
    var_6 = {var_3: var_5}
    var_7 = [var_2, var_6]
    var_8 = (var_1, var_7)
    var_9 = [var_0, var_8]
    var_10 = (var_4,)
    var_11 = {var_3: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_2, var_12]
    var_14 = module_1.freeze(var_9)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true_at_line_32. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = [var_5, var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 6/8 statements.
# Partially parsed test_freeze_complex_nesting. Retrieved 21/26 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_pvector_as_input. Retrieved 4/7 statements.


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
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
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

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

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
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = [var_4, var_9, var_12]
    var_14 = [var_1, var_2]
    var_15 = {var_6: var_7}
    var_16 = module_0.pmap(var_15)
    var_17 = (var_5, var_16)
    var_18 = {var_10, var_11}
    var_19 = module_1.pset(var_18)
    var_20 = module_2.freeze(var_13)

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_freeze_function_exists.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_strict_pmap_returns_pmap. Retrieved 6/9 statements.


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

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 12/26 statements.
# Partially parsed test_mutant_handles_kwargs_recursion. Retrieved 5/10 statements.
# Partially parsed test_mutant_preserves_logic_but_returns_frozen_structure. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]
    var_6 = 'a'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 'b'
    var_10 = {var_9: var_1}
    var_11 = 0

def test_case_0():
    var_0 = 'val'
    var_1 = 10
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 1

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_preserves_functionality_and_freezes_inputs. Retrieved 15/28 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = module_1.pset(var_5)
    var_7 = 3
    var_8 = [var_7]
    var_9 = {var_0: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_1, var_4]
    var_12 = module_1.pset(var_11)
    var_13 = [var_7]
    var_14 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_freeze_dict_nested. Retrieved 13/16 statements.
# Partially parsed test_freeze_list_simple. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_nested. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 7/9 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 8/11 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 18/26 statements.


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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

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
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = {var_4: var_5}
    var_11 = module_1.pmap(var_10)

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_1, var_2]

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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 12/21 statements.


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = [var_1, var_3]
    var_11 = {var_0: var_1}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_1, var_3]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = var_3[0]
    assert var_4 == 1

def test_case_0():
    var_0 = 'inner'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = var_4['inner'][0]
    assert var_5 == 1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'original'
    var_4 = {var_2: var_3}
    var_5 = [var_0]
    var_6 = var_4['key']
    assert var_6 == 'original'

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 2
    var_3 = (var_0, var_2)



# Parsed testcases at query #13
#--------------------------




import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.freeze(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_1.pmap(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_freezes_input_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_preserves_simple_types. Retrieved 1/6 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 13/26 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 42

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
    var_10 = (var_8, var_9)
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 8/13 statements.
# Partially parsed test_mutant_preserves_logic_but_returns_frozen_structure. Retrieved 4/9 statements.
# Partially parsed test_mutant_handles_nested_mutable_structures. Retrieved 9/18 statements.
# Failed to parse test_mutant_with_empty_inputs.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = module_0.pmap(var_6)

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
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_returns_frozen_value. Retrieved 7/15 statements.
# Partially parsed test_mutant_freezes_arguments. Retrieved 3/9 statements.
# Partially parsed test_mutant_preserves_functionality. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_of_ints. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_nested_tuple_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_strict_mode_default. Retrieved 8/13 statements.
# Partially parsed test_freeze_non_strict_mode_dict_keys. Retrieved 7/10 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 14/23 statements.


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
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = [var_5]

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'key'
    var_4 = 4
    var_5 = 5
    var_6 = [var_5]
    var_7 = (var_4, var_6)
    var_8 = {var_3: var_7}
    var_9 = [var_2, var_8]
    var_10 = (var_1, var_9)
    var_11 = [var_0, var_10]
    var_12 = [var_5]
    var_13 = module_0.freeze(var_11)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 13/23 statements.


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
    var_11 = [var_1, var_6]
    var_12 = [var_1, var_6]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 4/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 6/8 statements.
# Partially parsed test_freeze_list_of_dicts. Retrieved 13/14 statements.
# Partially parsed test_freeze_complex_structure. Retrieved 22/27 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.


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
    var_5 = {var_0, var_1, var_2}
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

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 'b'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = 6
    var_12 = 7
    var_13 = {var_11, var_12}
    var_14 = [var_0, var_5, var_10, var_13]
    var_15 = [var_2, var_3]
    var_16 = {var_7: var_8}
    var_17 = module_0.pmap(var_16)
    var_18 = (var_6, var_17)
    var_19 = {var_11, var_12}
    var_20 = module_1.pset(var_19)
    var_21 = module_2.freeze(var_14)

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_freezes_input_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 9/20 statements.
# Partially parsed test_mutant_with_multiple_args_and_kwargs. Retrieved 10/17 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = 4
    var_6 = [var_5]
    var_7 = {var_2: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 11/21 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 2/11 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 19/34 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = [var_0, var_1, var_2]
    var_9 = {var_4: var_5}
    var_10 = module_0.pmap(var_9)

def test_case_0():
    var_0 = 0
    var_1 = 1

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = 'inner'
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
    var_16 = 0
    var_17 = {var_12, var_13}
    var_18 = module_0.pset(var_17)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_predicate_evaluates_to_false. Retrieved 11/18 statements.


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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 11/20 statements.
# Partially parsed test_mutant_isolates_mutation_by_freezing_inputs. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 'val'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 'list'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = bool('new_key' not in var_2)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 8/15 statements.
# Partially parsed test_mutant_protects_against_internal_mutation. Retrieved 4/12 statements.
# Partially parsed test_mutant_recursive_freezing. Retrieved 7/17 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = bool(var_2 == [1, 2])
    assert var_4 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 'inner'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 9/20 statements.
# Partially parsed test_mutant_ensures_immutability_of_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_handles_kwargs_and_args_independently. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'x'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = 'key'
    var_8 = 'other'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'val'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = (var_5,)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_mutant_decorator_returns_function.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 11/23 statements.


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = 2
    var_10 = 3
    var_11 = [var_1, var_9, var_10]
    var_12 = [var_1, var_9, var_10]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




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

import pyrsistent._pvector as module_0
import pyrsistent._pmap as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = 4
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_1.m(**var_7)
    var_9 = [var_0, var_4, var_8]
    var_10 = module_0.v(*var_9)
    var_11 = module_2.thaw(var_10)
    var_12 = bool(var_11 == [1, [2, 3], {'a': 4}])
    assert var_12 is True

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
    var_5 = 'c'
    var_6 = {var_5: var_4}
    var_7 = module_1.m(**var_6)
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_3, var_9: var_7}
    var_11 = module_1.m(**var_10)
    var_12 = module_2.thaw(var_11)
    var_13 = bool(var_12 == {'a': [1, 2], 'b': {'c': 3}})
    assert var_13 is True

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

import pyrsistent._pvector as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.v(*var_3)
    var_5 = {var_0: var_4}
    var_6 = False
    var_7 = module_1.thaw(var_5, var_6)
    var_8 = [var_1, var_2]
    var_9 = module_0.v(*var_8)
    var_10 = {var_0: var_9}
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

import pyrsistent._pset as module_0
import pyrsistent._pvector as module_1
import pyrsistent._pmap as module_2
import pyrsistent._helpers as module_3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.s(*var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_5]
    var_7 = module_1.v(*var_6)
    var_8 = (var_4, var_7)
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_3, var_10: var_8}
    var_12 = module_2.m(**var_11)
    var_13 = 5
    var_14 = 'c'
    var_15 = {var_14: var_13}
    var_16 = module_2.m(**var_15)
    var_17 = [var_12, var_16]
    var_18 = module_1.v(*var_17)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_0, var_1}
    var_22 = [var_5]
    var_23 = (var_4, var_22)
    var_24 = {var_19: var_21, var_20: var_23}
    var_25 = 'c'
    var_26 = {var_25: var_13}
    var_27 = [var_24, var_26]
    var_28 = module_3.thaw(var_18)
    var_29 = bool(var_28 == var_27)
    assert var_29 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_list_to_pvector. Retrieved 16/24 statements.
# Partially parsed test_freeze_dict_to_pmap. Retrieved 17/20 statements.
# Partially parsed test_freeze_tuple_recursion. Retrieved 15/19 statements.
# Partially parsed test_freeze_nested_complex_structure. Retrieved 22/25 statements.


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
    var_9 = [var_2, var_3]
    var_10 = 'c'
    var_11 = 3
    var_12 = {var_10: var_11}
    var_13 = {var_0: var_9, var_1: var_12}
    var_14 = module_0.freeze(var_13)
    var_15 = [var_2, var_3]
    var_16 = {var_10: var_11}
    var_17 = module_1.pmap(var_16)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == (1, 2))
    assert var_4 is True
    var_5 = 3
    var_6 = [var_1, var_5]
    var_7 = (var_0, var_6)
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_5]
    var_10 = (var_0,)
    var_11 = [var_1]
    var_12 = (var_10, var_11)
    var_13 = module_0.freeze(var_12)
    var_14 = (var_0,)
    var_15 = [var_1]

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
    var_8 = (var_0, var_1)
    var_9 = {var_8, var_2}
    var_10 = module_0.freeze(var_9)
    var_11 = (var_0, var_1)
    var_12 = {var_11, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_4: var_3}
    var_8 = False
    var_9 = module_1.freeze(var_7, var_8)
    var_10 = {var_4: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = bool(var_9 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_2: var_5}
    var_7 = [var_3, var_4, var_6]
    var_8 = 4
    var_9 = 5
    var_10 = (var_8, var_9)
    var_11 = 6
    var_12 = 7
    var_13 = {var_11, var_12}
    var_14 = {var_0: var_7, var_1: var_10, var_2: var_13}
    var_15 = {var_2: var_5}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_3, var_4, var_16]
    var_18 = (var_8, var_9)
    var_19 = {var_11, var_12}
    var_20 = module_1.pset(var_19)
    var_21 = module_2.freeze(var_14)



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_freeze_function_exists.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_list. Retrieved 14/21 statements.
# Partially parsed test_freeze_dict. Retrieved 17/20 statements.
# Partially parsed test_freeze_tuple. Retrieved 15/19 statements.
# Partially parsed test_freeze_strict_false_dict_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_nested_complex. Retrieved 29/37 statements.
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
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True
    var_9 = [var_2, var_3]
    var_10 = 'c'
    var_11 = 3
    var_12 = {var_10: var_11}
    var_13 = {var_0: var_9, var_1: var_12}
    var_14 = module_0.freeze(var_13)
    var_15 = [var_2, var_3]
    var_16 = {var_10: var_11}
    var_17 = module_1.pmap(var_16)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == (1, 2))
    assert var_4 is True
    var_5 = 3
    var_6 = [var_1, var_5]
    var_7 = (var_0, var_6)
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_5]
    var_10 = (var_0,)
    var_11 = [var_1]
    var_12 = (var_10, var_11)
    var_13 = module_0.freeze(var_12)
    var_14 = (var_0,)
    var_15 = [var_1]

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
    var_8 = (var_0, var_1)
    var_9 = {var_8}
    var_10 = module_0.freeze(var_9)
    var_11 = (var_0, var_1)
    var_12 = [var_11]
    var_13 = module_1.pset(var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

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
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 'a'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = [var_4, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_10]
    var_12 = (var_9, var_11)
    var_13 = 5
    var_14 = 6
    var_15 = {var_13, var_14}
    var_16 = 'inner'
    var_17 = 7
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = {var_0: var_8, var_1: var_12, var_2: var_15, var_3: var_19}
    var_21 = {var_5: var_6}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_4, var_22]
    var_24 = [var_10]
    var_25 = [var_13, var_14]
    var_26 = module_1.pset(var_25)
    var_27 = [var_17]
    var_28 = module_2.freeze(var_20)

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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True
    var_8 = ()
    var_9 = module_0.freeze(var_8)
    var_10 = bool(var_9 == ())
    assert var_10 is True
    var_11 = set()
    var_12 = module_0.freeze(var_11)
    var_13 = []
    var_14 = module_2.pset(var_13)
    var_15 = bool(var_12 == var_14)
    assert var_15 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 7/16 statements.
# Partially parsed test_mutant_preserves_logic_but_converts_types. Retrieved 4/9 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_strict_pmap_is_true. Retrieved 7/10 statements.


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
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_2.type(*var_6, **var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 10/18 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 2/7 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 7/11 statements.


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
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_0]
    var_8 = {var_2: var_3}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 10
    var_1 = 'string'

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 'inner'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}



# Parsed testcases at query #9
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
    var_0 = 1.5
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == 1.5)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 8/18 statements.
# Partially parsed test_mutant_preserves_unrelated_mutation_isolation. Retrieved 5/11 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = [var_3]
    var_5 = 2
    var_6 = [var_1, var_5]
    var_7 = 'result'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_1, var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_decorator_returns_function. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 12/27 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_1, var_2]
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = [var_1, var_2]
    var_11 = [var_2, var_3]

def test_case_0():
    var_0 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #13
#--------------------------




import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_1.freeze(var_3)
    var_5 = module_0.pmap()
    var_6 = var_5.__class__
    var_7 = isinstance(var_4, var_6)
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_freeze_list_simple. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_nested. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 7/9 statements.


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
    var_6 = 'tuple_key'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_1.pmap(var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True
    var_10 = {var_0: var_2}
    var_11 = module_0.freeze(var_10)
    var_12 = {var_0: var_2}
    var_13 = module_1.pmap(var_12)
    var_14 = bool(var_11 == var_13)
    assert var_14 is True

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
    var_5 = {var_0, var_1, var_2}
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = (var_8, var_9)
    var_11 = [var_0, var_7, var_10]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = {var_3, var_4}
    var_6 = 'val'
    var_7 = 10
    var_8 = {var_6: var_7}
    var_9 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 7/17 statements.
# Partially parsed test_mutant_preserves_immutable_types. Retrieved 1/7 statements.
# Partially parsed test_mutant_deep_freezing_of_nested_structures. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = (var_5, var_7)
    var_9 = [var_4, var_8]
    var_10 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 8/19 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/8 statements.
# Partially parsed test_mutant_handles_empty_inputs. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1, var_2]
    var_7 = 0
    var_8 = bool(var_3 == [1, 2, 3])
    assert var_8 is True
    var_9 = bool(var_5 == {'a': 1})
    assert var_9 is True

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_strict_pmap_evaluates_true_at_line_32. Retrieved 6/10 statements.


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



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_mutant_decorator_returns_function.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 16/35 statements.
# Partially parsed test_mutant_preserves_functionality. Retrieved 4/9 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = 'a'
    var_6 = [var_0, var_1]
    var_7 = {var_5: var_6}
    var_8 = 'b'
    var_9 = 4
    var_10 = [var_9]
    var_11 = {var_8: var_10}
    var_12 = 0
    var_13 = 'key'
    var_14 = [var_0, var_1]
    var_15 = {var_13: var_14}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]

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
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = {var_0: var_6, var_1: var_10}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_decorator_preserves_functionality_and_freezes_inputs. Retrieved 13/24 statements.


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
    var_9 = [var_1, var_4, var_7]
    var_10 = [var_1, var_4, var_7]
    var_11 = 10
    var_12 = 20



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 6/15 statements.
# Partially parsed test_mutant_handles_empty_args. Retrieved 1/5 statements.
# Partially parsed test_mutant_handles_keyword_arguments. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = var_2['a']
    assert var_6 == 0
    var_7 = bool(var_4 == [1])
    assert var_7 is True

def test_case_0():
    var_0 = 10

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_mutant_decorator_returns_function.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_predicate_is_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_mutant_predicate_false.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_freezes_keyword_arguments. Retrieved 9/15 statements.
# Partially parsed test_mutant_deep_freezing. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'original'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = bool(var_2 == {'original': True})
    assert var_3 is True
    var_4 = 'mutated'
    var_5 = bool('mutated' not in var_2)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = var_6[0][0]
    assert var_8 == 1



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = bool(var_3 == [1, 2, 3])
    assert var_8 is True
    var_9 = bool(var_5 == {'a': 1})
    assert var_9 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_predicate_false. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_freezes_args_and_kwargs. Retrieved 11/20 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 5/12 statements.
# Partially parsed test_mutant_preserves_unmutable_types. Retrieved 3/9 statements.
# Partially parsed test_mutant_handles_nested_structures. Retrieved 3/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'x'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = [var_0, var_1, var_2]
    var_9 = {var_4: var_5}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)

def test_case_0():
    var_0 = 5
    var_1 = 'string'
    var_2 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 1



# Parsed testcases at query #32
#--------------------------




import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1



