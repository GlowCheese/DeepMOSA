####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/9 statements.
# Partially parsed test_freeze_tuple_with_elements. Retrieved 7/9 statements.
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
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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



# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = [var_5, var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 13/24 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 11/17 statements.
# Partially parsed test_mutant_with_positional_and_keyword_arguments. Retrieved 9/16 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_no_arguments. Retrieved 2/7 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/12 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_4 == {'a': 1, 'b': 2})
    assert var_7 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)
    var_6 = bool(var_3 == {1, 2, 3})
    assert var_6 is True

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'inner'
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = [var_2, var_3]
    var_12 = [var_6, var_7]
    var_13 = bool(var_10 == {'list': [1, 2], 'dict': {'inner': [3, 4]}})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_4}
    var_6 = [var_0]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = bool(var_0 == [])
    assert var_2 is True

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = (var_3, var_2)
    var_5 = [var_0, var_1]
    var_6 = bool(var_2 == [1, 2])
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple. Retrieved 12/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_nested_mixed. Retrieved 18/21 statements.
# Partially parsed test_freeze_already_frozen_strict. Retrieved 5/9 statements.
# Partially parsed test_freeze_already_frozen_not_strict. Retrieved 5/9 statements.


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
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = {var_1: var_2}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_4, var_5]

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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = False

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_empty_dict. Retrieved 3/4 statements.
# Partially parsed test_freeze_dict_with_values. Retrieved 9/11 statements.
# Partially parsed test_freeze_list. Retrieved 11/14 statements.
# Partially parsed test_freeze_set. Retrieved 5/6 statements.
# Partially parsed test_freeze_tuple. Retrieved 11/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 2/9 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/11 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 6/11 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 13/17 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_empty_set. Retrieved 3/4 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 3/4 statements.


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
    var_8 = var_7['a']
    assert var_8 == 1
    var_9 = var_7[var_1]
    var_10 = var_7['b'][0]
    assert var_10 == 2
    var_11 = var_7['b'][1]
    assert var_11 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = var_8[0]
    assert var_9 == 1
    var_10 = var_8[var_0]
    var_11 = var_8[1]['x']
    assert var_11 == 2
    var_12 = var_8[var_2]
    var_13 = var_8[2][0]
    assert var_13 == 3
    var_14 = var_8[2][1]
    assert var_14 == 4

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = 1
    var_6 = bool(1 in var_4)
    assert var_6 is True
    var_7 = 2
    var_8 = bool(2 in var_4)
    assert var_8 is True
    var_9 = 3
    var_10 = bool(3 in var_4)
    assert var_10 is True

import pyrsistent._helpers as module_0

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
    var_9 = var_8[0]
    assert var_9 == 1
    var_10 = var_8[var_0]
    var_11 = var_8[1][0]
    assert var_11 == 2
    var_12 = var_8[1][1]
    assert var_12 == 3
    var_13 = var_8[var_1]
    var_14 = var_8[2]['a']
    assert var_14 == 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1

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
    var_8 = bool(var_7 is var_5)
    assert var_8 is True

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
    var_8 = var_7[var_0]
    var_9 = var_7['a'][0]
    assert var_9 == 1
    var_10 = var_7['a'][1]
    assert var_10 == 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = var_9[var_0]
    var_11 = var_9[var_0][var_1]
    var_12 = var_9[var_0][var_1][var_2]
    var_13 = var_9['a']['b']['c'][0]
    assert var_13 == 1
    var_14 = var_9['a']['b']['c'][1]
    assert var_14 == 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/22 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = bool(var_3 == (1, 2, 3))
    assert var_4 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == {'x': 10})
    assert var_6 is True
    var_7 = bool(var_5 == {'y': 20})
    assert var_7 is True

def test_case_0():
    pass

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
    var_9 = bool(var_8 == {'list': [1, 2], 'set': {3, 4}})
    assert var_9 is True

def test_case_0():
    var_0 = 5
    assert var_0 == 5



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_keyword_arguments. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_handles_mixed_arguments. Retrieved 15/26 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 13/20 statements.
# Partially parsed test_mutant_decorator_with_no_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_decorator_freezes_returned_tuple. Retrieved 8/17 statements.
# Partially parsed test_mutant_decorator_with_already_frozen_arguments. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = [var_0, var_1, var_3]
    var_5 = bool(var_2 == [5, 6])
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_2 == {'a': 1})
    assert var_7 is True

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'x'
    var_7 = 5
    var_8 = {var_6: var_7}
    var_9 = [var_0, var_1]
    var_10 = module_0.pset(var_9)
    var_11 = [var_3, var_4]
    var_12 = {var_6: var_7}
    var_13 = module_1.pmap(var_12)
    var_14 = 0

def test_case_0():
    pass

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
    var_9 = 99
    var_10 = [var_9, var_3]
    var_11 = [var_5, var_6]
    var_12 = module_0.pset(var_11)
    var_13 = bool(var_8 == {'list': [1, 2], 'set': {3, 4}})
    assert var_13 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 17/22 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 10/11 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 9/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 3/10 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_13 = [var_3, var_4]
    var_14 = module_1.pset(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = module_1.freeze(var_4, var_5)
    var_7 = {var_0: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 == var_11)
    assert var_12 is True
    var_13 = [var_0, var_1, var_6]
    var_14 = 100
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'new_key'
    var_18 = {var_15: var_0, var_16: var_1, var_17: var_14}
    var_19 = module_0.m(**var_18)
    var_20 = 999
    var_21 = [var_0, var_1, var_6, var_20]



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 2/8 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 6/15 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 3/8 statements.
# Failed to parse test_mutant_decorator_with_empty_inputs.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = bool(var_2 == [1, 2])
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 10

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = bool(var_5 == {'list': [1, 2, 3]})
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = bool(var_2 == (1, 2))
    assert var_4 is True
    var_5 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = [var_0]
    var_10 = {var_2: var_3}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 12/14 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_freeze_already_frozen_pmap_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_already_frozen_pvector_strict. Retrieved 8/11 statements.
# Partially parsed test_freeze_already_frozen_pvector_non_strict. Retrieved 5/7 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.


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
    var_13 = {var_6, var_7}
    var_14 = module_1.pset(var_13)
    var_15 = {var_5: var_14}
    var_16 = module_2.pmap(var_15)

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = True
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = False
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = bool(var_7 is var_5)
    assert var_8 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = False

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 12/23 statements.
# Partially parsed test_mutant_decorator_with_empty_arguments. Retrieved 2/10 statements.
# Partially parsed test_mutant_decorator_freezes_set. Retrieved 7/13 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 5/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == [1, 2, 3])
    assert var_6 is True
    var_7 = bool(var_5 == {'a': 1})
    assert var_7 is True
    var_8 = 0
    var_9 = 4
    var_10 = [var_0, var_1, var_2, var_9]
    var_11 = 'new'
    var_12 = 5
    var_13 = {var_4: var_0, var_11: var_12}
    var_14 = module_0.pmap(var_13)

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = bool(var_1 == [1])
    assert var_3 is True
    var_4 = [var_0, var_2]

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
    var_10 = bool(var_9 == {'list': [1, 2, 3], 'tuple': (4, 5)})
    assert var_10 is True
    var_11 = 99
    var_12 = [var_11, var_3, var_4]

def test_case_0():
    var_0 = 'empty'
    var_1 = []

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = bool(var_3 == {1, 2, 3})
    assert var_4 is True
    var_5 = 4
    var_6 = {var_0, var_1, var_2, var_5}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = {}
    var_1 = bool(var_0 == {})
    assert var_1 is True
    var_2 = 'inner'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 6/7 statements.
# Partially parsed test_mutant_with_mutable_input. Retrieved 9/10 statements.
# Partially parsed test_mutant_with_no_args. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_strict_freeze. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_dict_keys_not_frozen. Retrieved 10/14 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: [x, y]
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = var_1(var_2, var_3)
    var_5 = [var_2, var_3]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = lambda a, b: {a: b}
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = var_1(var_2, var_3)
    var_5 = {var_2: var_3}
    var_6 = module_1.pmap(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda lst: lst.append(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2(var_6)
    var_8 = [var_3, var_4, var_5, var_0]

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
    var_10 = var_4(var_9)
    var_11 = {var_5: var_7, var_6: var_8, var_0: var_1}
    var_12 = module_1.pmap(var_11)
    var_13 = bool(var_10 == var_12)
    assert var_13 is True

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
    var_8 = var_3(var_7)
    var_9 = {var_4, var_5, var_6, var_0}
    var_10 = module_1.pset(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

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
    var_8 = var_3(var_7)
    var_9 = bool(var_8 == (1, 2, 3, 4))
    assert var_9 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = lambda x, y, z: {var_0: x, var_1: y, var_2: z}
    var_4 = module_0.mutant(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = var_4(var_5, var_6, var_7)
    var_9 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_10 = module_1.pmap(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = var_5()
    var_7 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda v: v.append(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5, var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = hash(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = lambda d: d
    var_6 = module_0.mutant(var_5)
    var_7 = 'value'
    var_8 = {var_2: var_7}
    var_9 = var_6(var_8)
    var_10 = 0



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 12/22 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/23 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 2/5 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = bool(var_2 == (1, 2))
    assert var_4 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == {'x': 10})
    assert var_6 is True
    var_7 = bool(var_5 == {'y': 20})
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'b'
    var_9 = 5
    var_10 = bool(var_2 == [1, 2])
    assert var_10 is True
    var_11 = bool(var_7 == {'a': [3, 4]})
    assert var_11 is True
    var_12 = [var_4, var_5]
    var_13 = [var_0, var_1, var_9]

def test_case_0():
    pass

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
    var_9 = bool(var_8 == {'list': [1, 2], 'set': {3, 4}})
    assert var_9 is True

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple_with_elements. Retrieved 12/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 22/27 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 9/12 statements.


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

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'x'
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
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 2
    var_4 = 'b'
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = {var_4: var_7}
    var_9 = [var_2, var_3, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = [var_11]
    var_13 = (var_10, var_12)
    var_14 = {var_0: var_9, var_1: var_13}
    var_15 = module_0.freeze(var_14)
    var_16 = [var_5, var_6]
    var_17 = module_1.pset(var_16)
    var_18 = {var_4: var_17}
    var_19 = module_2.pmap(var_18)
    var_20 = [var_2, var_3, var_19]
    var_21 = [var_11]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple. Retrieved 12/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false. Retrieved 10/11 statements.
# Partially parsed test_freeze_strict_true. Retrieved 9/14 statements.


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
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_3, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = {var_1: var_2}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_4, var_5]

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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = module_1.freeze(var_4, var_5)
    var_7 = {var_0: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_4]
    var_6 = True
    var_7 = module_1.freeze(var_5, var_6)
    var_8 = [var_6]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = {var_2: var_3}
    var_9 = module_1.pmap(var_8)
    var_10 = {var_1: var_9}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_1.pmap(var_12)
    var_14 = bool(var_7 == var_13)
    assert var_14 is True



# Parsed testcases at query #31
#--------------------------




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
    var_8 = bool(var_7 is var_5)
    assert var_8 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/9 statements.
# Partially parsed test_freeze_tuple_with_elements. Retrieved 11/13 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 3/10 statements.
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
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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



# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True



# Parsed testcases at query #35
#--------------------------






# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_with_positional_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 11/18 statements.
# Partially parsed test_mutant_return_value_frozen. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_nested_mutables. Retrieved 12/18 statements.
# Partially parsed test_mutant_no_arguments. Retrieved 4/8 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = [var_1, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = bool(var_4 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = {var_0: var_2, var_1: var_3, var_5: var_6}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 5
    var_7 = bool(var_2 == [1, 2])
    assert var_7 is True
    var_8 = bool(var_5 == {'x': 10})
    assert var_8 is True
    var_9 = [var_0, var_1, var_6]
    var_10 = 'factor'
    var_11 = {var_3: var_4, var_10: var_6}
    var_12 = module_0.pmap(var_11)

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
    var_0 = 'first'
    var_1 = 'second'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(var_4 == {'first': 'a', 'second': 'b'})
    assert var_9 is True
    var_10 = bool(var_8 == [1, 2, 3])
    assert var_10 is True
    var_11 = {var_0: var_5, var_1: var_3}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_2, var_6, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'answer'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #38
#--------------------------






# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 5/8 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 9/14 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 13/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_8 = True
    var_9 = module_1.freeze(var_2, var_8)
    var_10 = {var_1: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = [var_6, var_11]

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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []

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
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 10/15 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 9/12 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/13 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_mutable_structures. Retrieved 10/26 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 7/13 statements.
# Partially parsed test_mutant_decorator_return_frozen. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = bool(var_2 == [1, 2])
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == {'x': 10})
    assert var_6 is True
    var_7 = bool(var_5 == {'y': 20})
    assert var_7 is True

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
    var_10 = bool(var_9 == {'list': [1, 2], 'tuple': (3, [4])})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 1



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_thaw_pvector. Retrieved 4/7 statements.
# Partially parsed test_thaw_pvector_nested. Retrieved 6/9 statements.
# Partially parsed test_thaw_pmap_nested. Retrieved 5/9 statements.
# Partially parsed test_thaw_pset. Retrieved 4/7 statements.
# Partially parsed test_thaw_tuple. Retrieved 5/9 statements.
# Partially parsed test_thaw_strict_false_list. Retrieved 4/7 statements.
# Partially parsed test_thaw_mixed_nested. Retrieved 10/15 statements.
# Partially parsed test_thaw_empty_containers. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = [var_0, var_7]

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
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_1}
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = [var_0, var_1]
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = (var_0, var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = [var_0, var_1]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = False
    var_5 = module_1.thaw(var_3, var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.thaw(var_2, var_3)
    var_5 = [var_3, var_1]
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.thaw(var_2, var_3)
    var_5 = {var_0: var_3}
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = 'a'
    var_7 = {var_0, var_1}
    var_8 = {var_6: var_7}
    var_9 = (var_3, var_4)
    var_10 = [var_8, var_9]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = []
    var_3 = {}
    var_4 = module_0.m(**var_3)
    var_5 = module_1.thaw(var_4, var_1)
    var_6 = {}
    var_7 = bool(var_5 == var_6)
    assert var_7 is True
    var_8 = []
    var_9 = set()
    var_10 = bool(var_5 == var_9)
    assert var_10 is True
    var_11 = ()
    var_12 = module_1.thaw(var_11, var_1)
    var_13 = ()
    var_14 = bool(var_12 == var_13)
    assert var_14 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_thaw_pset_converts_to_set. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0, var_1, var_2}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_with_positional_args. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_keyword_args. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_multiple_args. Retrieved 8/9 statements.
# Partially parsed test_mutant_with_args_and_kwargs. Retrieved 8/9 statements.
# Partially parsed test_mutant_return_frozen_dict. Retrieved 8/11 statements.
# Partially parsed test_mutant_return_frozen_tuple. Retrieved 7/9 statements.
# Partially parsed test_mutant_input_frozen. Retrieved 8/9 statements.
# Partially parsed test_mutant_input_dict_frozen. Retrieved 9/12 statements.
# Partially parsed test_mutant_nested_input_frozen. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_defaultdict. Retrieved 8/14 statements.
# Failed to parse test_mutant_preserves_function_name.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = lambda x: x + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_3(var_5)
    var_7 = [var_4, var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = lambda x=[]: x + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_3(x=var_5)
    var_7 = [var_4, var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = var_1(var_3, var_5)
    var_7 = [var_2, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = var_1(var_3, y=var_5)
    var_7 = [var_2, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = lambda : var_3
    var_5 = module_0.mutant(var_4)
    var_6 = var_5()
    var_7 = [var_1]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = lambda : var_2
    var_4 = module_0.mutant(var_3)
    var_5 = var_4()
    var_6 = [var_0, var_1]
    var_7 = module_1.pset(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = (var_1,)
    var_3 = lambda : var_2
    var_4 = module_0.mutant(var_3)
    var_5 = var_4()
    var_6 = [var_0]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = lambda x: x.append(var_3) or x
    var_5 = module_0.mutant(var_4)
    var_6 = var_5(var_2)
    var_7 = [var_0, var_1, var_3]
    var_8 = bool(var_2 == [1, 2])
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 2
    var_5 = lambda d: d[var_0].append(var_4) or d
    var_6 = module_0.mutant(var_5)
    var_7 = var_6(var_3)
    var_8 = [var_1, var_4]
    var_9 = bool(var_3 == {'a': [1]})
    assert var_9 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = lambda s: s.add(var_3) or s
    var_5 = module_0.mutant(var_4)
    var_6 = var_5(var_2)
    var_7 = [var_0, var_1, var_3]
    var_8 = module_1.pset(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True
    var_10 = bool(var_2 == {1, 2})
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = lambda x: x[var_0].append(var_5) or x
    var_7 = module_0.mutant(var_6)
    var_8 = var_7(var_4)
    var_9 = [var_1, var_2, var_5]
    var_10 = bool(var_4 == {'list': [1, 2]})
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 2
    var_5 = lambda d: d[var_0].append(var_4) or d
    var_6 = module_0.mutant(var_5)
    var_7 = [var_1, var_4]



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/10 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 22/27 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 4/6 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_0, var_1]

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
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = set(var_6)
    var_8 = {var_3: var_7}
    var_9 = [var_2, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = [var_11]
    var_13 = (var_10, var_12)
    var_14 = {var_0: var_9, var_1: var_13}
    var_15 = module_0.freeze(var_14)
    var_16 = [var_4, var_5]
    var_17 = module_1.pset(var_16)
    var_18 = {var_3: var_17}
    var_19 = module_2.pmap(var_18)
    var_20 = [var_2, var_19]
    var_21 = [var_11]

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
    var_8 = bool(var_7 is var_5)
    assert var_8 is True

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



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_freeze_dict_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 5/7 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/10 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 4/6 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 22/27 statements.


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
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
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
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_0, var_1]

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
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

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

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = set(var_6)
    var_8 = {var_3: var_7}
    var_9 = [var_2, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = [var_11]
    var_13 = (var_10, var_12)
    var_14 = {var_0: var_9, var_1: var_13}
    var_15 = module_0.freeze(var_14)
    var_16 = [var_4, var_5]
    var_17 = module_1.pset(var_16)
    var_18 = {var_3: var_17}
    var_19 = module_2.pmap(var_18)
    var_20 = [var_2, var_19]
    var_21 = [var_11]



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 10/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_dict. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_false_with_list. Retrieved 12/16 statements.
# Partially parsed test_freeze_mixed_structure. Retrieved 25/31 statements.


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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0, var_1]
    var_9 = [var_3, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = (var_0, var_1, var_2)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = [var_0, var_1]
    var_9 = (var_3, var_4)

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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = module_0.freeze(var_6, var_5)
    var_8 = [var_1, var_2]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = module_0.freeze(var_6, var_7)
    var_9 = module_0.freeze(var_8, var_7)
    var_10 = [var_0, var_1]
    var_11 = [var_3, var_4]

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = 'x'
    var_14 = 7
    var_15 = 8
    var_16 = [var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = {var_0: var_6, var_1: var_9, var_2: var_12, var_3: var_17}
    var_19 = module_0.freeze(var_18)
    var_20 = [var_4, var_5]
    var_21 = (var_7, var_8)
    var_22 = [var_10, var_11]
    var_23 = module_1.pset(var_22)
    var_24 = [var_14, var_15]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 9/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_strict_true. Retrieved 10/15 statements.


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
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_13 = {var_6, var_7}
    var_14 = module_1.pset(var_13)
    var_15 = {var_5: var_14}
    var_16 = module_2.pmap(var_15)

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

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






