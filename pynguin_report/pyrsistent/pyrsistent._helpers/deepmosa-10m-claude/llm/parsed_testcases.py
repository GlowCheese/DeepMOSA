####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_defaultdict_with_nested_values. Retrieved 2/7 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 5/9 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/9 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': {'b': 1}})
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = bool(var_6 == {'a': [1, 2, 3]})
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

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
    var_8 = bool(var_7 == [[1, 2], [3, 4]])
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == [1, {'a': 2}])
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = set()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == {1, 2, 3})
    assert var_5 is True

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
    var_6 = bool(var_5 == (1, [2, 3]))
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == (1, {'a': 2}))
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = {var_3: var_6}
    var_8 = [var_2, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_8, var_1: var_11}
    var_13 = module_0.freeze(var_12)
    var_14 = bool(var_13 == {'a': [1, {'b': (2, 3)}], 'c': {4, 5}})
    assert var_14 is True

def test_case_0():
    var_0 = 1
    var_1 = 2

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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = True
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = bool(var_5 is var_3)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = bool(var_7 == {'a': {'b': {'c': 1}}})
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = module_0.freeze(var_11)
    var_13 = bool(var_12 == [[[1, 2], [3, 4]], [[5, 6]]])
    assert var_13 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'z'
    var_2 = 1
    var_3 = 2
    var_4 = 'y'
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_4: var_7}
    var_9 = [var_2, var_3, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = {var_0: var_9, var_1: var_12}
    var_14 = module_0.freeze(var_13)
    var_15 = bool(var_14 == {'x': [1, 2, {'y': (3, 4)}], 'z': {5, 6}})
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_freeze_defaultdict.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True
    var_3 = len(var_1)
    assert var_3 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = var_5['a']
    assert var_6 == 1
    var_7 = var_5['b']
    assert var_7 == 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = var_5['a']['b']
    assert var_6 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 2
    var_7 = var_4[2]
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1][0]
    assert var_7 == 2
    var_8 = var_5[1][1]
    assert var_8 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]['a']
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

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
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 2
    var_7 = var_4[2]
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1][0]
    assert var_7 == 2
    var_8 = var_5[1][1]
    assert var_8 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = var_3[0]
    assert var_4 == 1
    var_5 = var_3[var_0]
    var_6 = len(var_5)
    assert var_6 == 0

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 2
    var_6 = 'nested'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = 6
    var_13 = [var_11, var_12]
    var_14 = (var_10, var_13)
    var_15 = 7
    var_16 = 8
    var_17 = {var_15, var_16}
    var_18 = 'inner'
    var_19 = 9
    var_20 = {var_18: var_19}
    var_21 = {var_0: var_9, var_1: var_14, var_2: var_17, var_3: var_20}
    var_22 = module_0.freeze(var_21)
    var_23 = var_22['list'][0]
    assert var_23 == 1
    var_24 = var_22['list'][2]['nested']
    assert var_24 == 3
    var_25 = var_22['tuple'][0]
    assert var_25 == 4
    var_26 = var_22['tuple'][1][0]
    assert var_26 == 5
    var_27 = 7
    var_28 = bool(7 in var_22['set'])
    assert var_28 is True
    var_29 = var_22['dict']['inner']
    assert var_29 == 9

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = var_5['a'][0]
    assert var_6 == 1
    var_7 = var_5['a'][1]
    assert var_7 == 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0][0]
    assert var_6 == 1
    var_7 = var_5[1]
    assert var_7 == 3



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 7/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap()
    var_7 = [var_6]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 8/15 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_set. Retrieved 6/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 9/14 statements.
# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_empty_dict. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_set. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 2/4 statements.
# Partially parsed test_freeze_strict_true. Retrieved 7/15 statements.
# Partially parsed test_freeze_strict_false. Retrieved 5/10 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True
    var_7 = [var_5]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': {'b': 1}})
    assert var_6 is True
    var_7 = [var_5]
    var_8 = var_5[var_0]
    var_9 = [var_8]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True
    var_7 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, [2, 3]])
    assert var_7 is True
    var_8 = [var_5]
    var_9 = var_5[var_0]
    var_10 = [var_9]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]
    var_8 = bool(var_5[1] == {'a': 3})
    assert var_8 is True
    var_9 = [var_5]
    var_10 = var_5[var_0]
    var_11 = [var_10]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = 1
    var_7 = bool(1 in var_5)
    assert var_7 is True
    var_8 = 2
    var_9 = bool(2 in var_5)
    assert var_9 is True
    var_10 = 3
    var_11 = bool(3 in var_5)
    assert var_11 is True
    var_12 = [var_5]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True
    var_6 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[var_0]
    var_8 = list(var_7)
    var_9 = bool(var_8 == [2, 3])
    assert var_9 is True
    var_10 = [var_5]
    var_11 = var_5[var_0]
    var_12 = [var_11]

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True
    var_3 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True
    var_3 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

import pyrsistent._helpers as module_0

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
    var_13 = module_0.freeze(var_12)
    var_14 = var_13['a'][1]
    var_15 = bool(var_13['a'][1] == {'b': 2})
    assert var_15 is True
    var_16 = var_13['c'][1][0]
    assert var_16 == 4



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/9 statements.
# Failed to parse test_mutant_return_value_is_frozen.
# Partially parsed test_mutant_with_empty_containers. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = {var_2: var_0}

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = []
    var_1 = {}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 11/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = True
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = [var_5, var_1, var_2]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 7/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap()
    var_7 = [var_6]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_dict_with_list. Retrieved 8/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/5 statements.
# Partially parsed test_freeze_simple_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/11 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 9/11 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/12 statements.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 7/12 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 31/38 statements.
# Partially parsed test_freeze_strict_true_pmap. Retrieved 9/11 statements.
# Partially parsed test_freeze_strict_true_pvector. Retrieved 8/11 statements.
# Partially parsed test_freeze_list_of_dicts. Retrieved 13/15 statements.
# Partially parsed test_freeze_dict_with_tuple_values. Retrieved 8/13 statements.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = [var_1, var_2, var_3]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)
    var_5 = {var_2: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'set'
    var_3 = 'tuple'
    var_4 = 1
    var_5 = 2
    var_6 = 'nested'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 'inner'
    var_11 = 4
    var_12 = 5
    var_13 = [var_11, var_12]
    var_14 = {var_10: var_13}
    var_15 = 6
    var_16 = 7
    var_17 = {var_15, var_16}
    var_18 = 8
    var_19 = 9
    var_20 = [var_19]
    var_21 = (var_18, var_20)
    var_22 = {var_0: var_9, var_1: var_14, var_2: var_17, var_3: var_21}
    var_23 = module_0.freeze(var_22)
    var_24 = {var_6: var_7}
    var_25 = module_1.pmap(var_24)
    var_26 = [var_4, var_5, var_25]
    var_27 = var_23['list']
    var_28 = [var_11, var_12]
    var_29 = var_23['dict']
    var_30 = [var_15, var_16]
    var_31 = module_2.pset(var_30)
    var_32 = var_23['set']
    var_33 = bool(var_23['set'] == var_31)
    assert var_33 is True
    var_34 = [var_19]
    var_35 = var_23['tuple']

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
    var_8 = var_7[var_0]

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
    var_6 = module_0.freeze(var_5)
    var_7 = [var_2]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 10/15 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 5/11 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = [var_6, var_7]
    var_9 = module_0.freeze(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}

def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 6/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/19 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/16 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 10/16 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_0 = 'list'
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'test'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 4
    var_9 = [var_2, var_3, var_4, var_8]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.pset(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_3: var_4}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = {}
    var_1 = 'outer'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/15 statements.
# Failed to parse test_mutant_return_value_is_frozen.
# Partially parsed test_mutant_with_primitive_types. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

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
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'list'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'a'

def test_case_0():
    pass

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 1
    var_3 = 2
    var_4 = 'inner'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'list'
    var_4 = 'dict'
    var_5 = 'set'

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/10 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_with_dict_argument. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/14 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_nested_list_in_dict. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = 'new_key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = module_0.pmap()
    var_6 = [var_5]

def test_case_0():
    pass

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]

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
    var_4 = 4
    var_5 = [var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 6/15 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/22 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/18 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/22 statements.
# Partially parsed test_mutant_with_scalar_return. Retrieved 1/5 statements.
# Partially parsed test_mutant_deeply_nested_structures. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 999
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
    var_0 = 'items'
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
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'b_key'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 'list'
    var_6 = 99
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    pass

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = {}
    var_1 = 'nested'
    var_2 = 'list'
    var_3 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 12/21 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 11/19 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 10/19 statements.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 10/17 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = 999
    var_6 = [var_0, var_1, var_2, var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = bool(var_7 == [[1, 2, 3], {'key': 'value'}])
    assert var_8 is True
    var_9 = 5
    var_10 = [var_0, var_1, var_2, var_9]
    var_11 = {var_4: var_5}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 10
    var_6 = [var_0, var_1, var_5]
    var_7 = 'new_key'
    var_8 = 'new_value'
    var_9 = {var_3: var_0, var_7: var_8}
    var_10 = module_0.pmap(var_9)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = bool(var_3 == {1, 2, 3})
    assert var_4 is True
    var_5 = 999
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = bool(var_4 == (1, [2, 3]))
    assert var_5 is True
    var_6 = [var_1, var_2]
    var_7 = 999

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 999
    var_7 = [var_0, var_1, var_6]
    var_8 = 888
    var_9 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = bool(var_7 == {'nested': {'list': [1, 2, 3]}})
    assert var_8 is True
    var_9 = 100
    var_10 = [var_2, var_3, var_4, var_9]

def test_case_0():
    var_0 = 5
    var_1 = 3



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 5/11 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 15/36 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = [var_2, var_3, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = [var_2, var_3, var_6]
    var_10 = module_2.freeze(var_9)
    var_11 = [var_10]
    var_12 = 'x'
    var_13 = 10
    var_14 = {var_12: var_13}
    var_15 = module_2.freeze(var_14)
    var_16 = [var_15]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/25 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_deeply_nested_structures. Retrieved 10/27 statements.
# Failed to parse test_mutant_return_value_is_frozen.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 'list'
    var_8 = 'dict'
    var_9 = module_0.pmap()
    var_10 = [var_9]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0.pmap()
    var_5 = [var_4]

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap()
    var_9 = [var_8]
    var_10 = module_0.pmap()
    var_11 = [var_10]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/16 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 10/18 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 10/17 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 10/18 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = module_0.freeze(var_4)
    var_6 = [var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.freeze(var_7)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_7]
    var_9 = {var_0: var_1}
    var_10 = module_0.freeze(var_9)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = module_0.freeze(var_4)
    var_6 = [var_5]
    var_7 = 4
    var_8 = 6
    var_9 = [var_1, var_7, var_8]
    var_10 = module_0.freeze(var_9)

def test_case_0():
    pass

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.freeze(var_5)
    var_7 = [var_6]
    var_8 = {var_1: var_2}
    var_9 = {var_0: var_8}
    var_10 = module_0.freeze(var_9)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 5/15 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(var_4 == {'a': 1, 'b': 2})
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 4/8 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 10/18 statements.
# Failed to parse test_mutant_with_primitive_return.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 10/20 statements.
# Partially parsed test_mutant_multiple_arguments. Retrieved 11/21 statements.


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
    var_4 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'nested'
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
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = 'y'
    var_6 = {var_5: var_1}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = [var_1, var_4, var_6]
    var_8 = 'd1'
    var_9 = 'd2'
    var_10 = 'l'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/8 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/5 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 4/11 statements.
# Partially parsed test_mutant_return_value_is_frozen. Retrieved 1/5 statements.
# Partially parsed test_mutant_with_none_argument. Retrieved 1/4 statements.
# Partially parsed test_mutant_with_primitive_types. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_defaultdict_argument. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'val'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 3
    var_1 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'append'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/14 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 14/28 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/11 statements.
# Partially parsed test_mutant_prevents_mutation_of_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 8/23 statements.


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
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_14]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = []
    var_5 = module_0.pset(var_4)
    var_6 = [var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = []

def test_case_0():
    pass

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = []
    var_2 = 'dict'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_4]
    var_6 = 'set'
    var_7 = []
    var_8 = module_1.pset(var_7)
    var_9 = [var_8]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/19 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_mixed_arguments_and_kwargs. Retrieved 7/17 statements.
# Failed to parse test_mutant_with_scalar_return_value.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = [var_4]

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
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_1, var_2, var_4, var_5]

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
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 'list'
    var_7 = 'dict'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'list'
    var_4 = 'dict'
    var_5 = 'set'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/8 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 8/19 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Failed to parse test_mutant_with_no_arguments.
# Failed to parse test_mutant_freezes_return_value_list.
# Failed to parse test_mutant_freezes_return_value_dict.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'initial'
    var_1 = 'data'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'existing'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 'set'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = (var_0, var_3, var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = set()
    var_3 = 'dict'
    var_4 = 'list'
    var_5 = 'set'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = True
    var_5 = module_0.pmap()
    var_6 = [var_5]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 12/20 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/20 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_primitive_types. Retrieved 3/7 statements.


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
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = [var_2, var_3]
    var_10 = {var_5: var_6}
    var_11 = module_0.pmap(var_10)

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'nested'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_4, var_1]
    var_9 = True
    var_10 = {var_3: var_9}
    var_11 = module_0.pmap(var_10)

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
    var_4 = 4
    var_5 = (var_0, var_3, var_4)
    var_6 = [var_1, var_2]

def test_case_0():
    pass

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = []
    var_7 = module_1.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 14/32 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = '__hash__'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_9]
    var_11 = [var_0, var_1]
    var_12 = 'x'
    var_13 = 10
    var_14 = {var_12: var_13}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/13 statements.
# Partially parsed test_mutant_decorator_with_persistent_structures. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 4/9 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 'nested'
    var_2 = 'dict'
    var_3 = {var_1: var_2}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 3/15 statements.
# Partially parsed test_mutant_with_nested_list_and_dict. Retrieved 3/13 statements.


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
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'result'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

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
    var_3 = 'key'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    pass

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'set'

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'nested'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/15 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 7/24 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 2/13 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/24 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/14 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 15/32 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 'new_key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = [var_0]
    var_2 = 'y'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = 'b'

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = 4

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
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 5
    var_7 = 6
    var_8 = {var_6, var_7}
    var_9 = module_0.pmap()
    var_10 = [var_9]
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = module_0.pmap()
    var_14 = [var_13]
    var_15 = 'set'
    var_16 = module_1.pset()
    var_17 = [var_16]

def test_case_0():
    pass



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 15/32 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = [var_3, var_4]
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = [var_9]
    var_11 = [var_2, var_3, var_4]
    var_12 = [var_2, var_3, var_4]
    var_13 = module_0.freeze(var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.freeze(var_15)
    var_17 = [var_16]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate_line_32. Retrieved 14/21 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_5, var_3]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = {var_6: var_8, var_7: var_11}
    var_13 = True
    var_14 = 'x'
    var_15 = 'y'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 12/20 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 7/13 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 11/19 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/17 statements.
# Partially parsed test_mutant_with_mixed_kwargs_and_args. Retrieved 16/24 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_empty_structures. Retrieved 8/16 statements.


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
    var_1 = 2
    var_2 = 'x'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_1, var_5]

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
    var_4 = 4
    var_5 = (var_0, var_3, var_4)
    var_6 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

def test_case_0():
    var_0 = 'documented function'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_0]
    var_12 = {var_2: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_5: var_6}
    var_15 = module_0.pmap(var_14)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = []
    var_7 = module_1.pset(var_6)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/9 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/10 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 1/7 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/14 statements.
# Failed to parse test_mutant_with_primitive_return.
# Partially parsed test_mutant_deeply_nested_structures. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 'key'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'list'
    var_7 = 'dict'

def test_case_0():
    var_0 = 5
    var_1 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 'set'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = 'tuple'

def test_case_0():
    pass

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/19 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 9/21 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/18 statements.


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
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = [var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = (var_0, var_3, var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'list'
    var_7 = 'dict'

def test_case_0():
    pass

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'list'
    var_4 = 'dict'
    var_5 = 'set'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 3/6 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/14 statements.
# Partially parsed test_mutant_freezes_multiple_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_empty_collections. Retrieved 3/8 statements.
# Partially parsed test_mutant_return_value_is_frozen. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()

def test_case_0():
    var_0 = 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/16 statements.
# Partially parsed test_mutant_deeply_nested_structures. Retrieved 10/20 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'result'
    var_7 = 'data'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'list'
    var_4 = 'dict'
    var_5 = 'set'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_dict_and_list. Retrieved 8/14 statements.
# Failed to parse test_mutant_return_value_is_frozen.


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
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

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
    var_4 = {var_3: var_0}

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = {var_2: var_0}

def test_case_0():
    pass

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_freeze_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list. Retrieved 9/11 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/12 statements.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 7/12 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 25/31 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/5 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 6/10 statements.
# Partially parsed test_freeze_list_of_dicts. Retrieved 13/15 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 12/17 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_1.pset(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'
    var_4 = 3.14
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == 3.14)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    assert var_1 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = 'a'
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)
    var_5 = {var_2: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'tuple'
    var_3 = 1
    var_4 = 2
    var_5 = 'nested'
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
    var_18 = module_0.freeze(var_17)
    var_19 = {var_5: var_6}
    var_20 = module_1.pmap(var_19)
    var_21 = [var_3, var_4, var_20]
    var_22 = [var_9, var_10]
    var_23 = module_2.pset(var_22)
    var_24 = [var_13, var_14]

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
    var_2 = []
    var_3 = module_1.pset(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = [var_4, var_1, var_2]

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = [var_2, var_3]
    var_11 = [var_5, var_6]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 9/26 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_1, var_6: var_2}
    var_8 = {var_1, var_2, var_3}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_multiple_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_deeply_nested_structures. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = 99
    var_6 = [var_0, var_1, var_2, var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = bool(var_5 == {'items': [1, 2, 3]})
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = bool(var_3 == {1, 2, 3})
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = bool(var_7 == {'level1': {'level2': [1, 2, 3]}})
    assert var_8 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 9/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = var_0[0].__name__
    assert var_6 == 'PMap'
    var_7 = '__hash__'
    var_8 = {var_1: var_3, var_2: var_4}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 9/23 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/25 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/13 statements.
# Partially parsed test_mutant_preserves_set. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/18 statements.
# Partially parsed test_mutant_with_scalar_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_none_return. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_5]
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = [var_8]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = [var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'original'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = 'modified'

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = (var_0, var_3, var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/14 statements.
# Partially parsed test_mutant_decorator_with_list_argument. Retrieved 4/14 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 3/11 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = bool(var_4 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = 'new_key'
    var_9 = bool('new_key' not in var_4)
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.pmap()
    var_3 = [var_2]

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'original'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = bool(var_4 == {'nested': {'key': 'original'}})
    assert var_7 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_evaluates_to_false. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 12/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 15/24 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 11/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/9 statements.
# Partially parsed test_mutant_deep_nesting. Retrieved 13/21 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 7/13 statements.


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

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'x'
    var_7 = 5
    var_8 = {var_6: var_7}
    var_9 = 'lists'
    var_10 = 'dict'
    var_11 = [var_0, var_1]
    var_12 = [var_3, var_4]
    var_13 = {var_6: var_7}
    var_14 = module_0.pmap(var_13)

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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_4: var_5}
    var_11 = module_0.pmap(var_10)
    var_12 = [var_2, var_3, var_11]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'set'
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = module_1.pset()



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/17 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 4/8 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

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
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    pass

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'existing'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'data'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_thaw_pvector_to_list. Retrieved 3/7 statements.
# Partially parsed test_thaw_pmap_to_dict. Retrieved 4/6 statements.
# Partially parsed test_thaw_pset_to_set. Retrieved 3/7 statements.
# Partially parsed test_thaw_tuple_recursive. Retrieved 3/8 statements.
# Partially parsed test_thaw_nested_pvector_pmap. Retrieved 3/6 statements.
# Partially parsed test_thaw_nested_pmap_pvector. Retrieved 2/6 statements.
# Failed to parse test_thaw_empty_pvector.
# Partially parsed test_thaw_empty_pset. Retrieved 1/4 statements.
# Partially parsed test_thaw_deeply_nested_structures. Retrieved 3/8 statements.
# Partially parsed test_thaw_tuple_with_pset. Retrieved 3/7 statements.
# Partially parsed test_thaw_pmap_with_nested_tuple. Retrieved 3/8 statements.
# Partially parsed test_thaw_list_with_none_elements. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = module_1.thaw(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = set()

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.thaw(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    assert var_1 == 42
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = 3.14
    var_5 = module_0.thaw(var_4)
    var_6 = bool(var_5 == 3.14)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.thaw(var_7)
    assert var_8 is None

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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
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
    var_5 = False
    var_6 = module_0.thaw(var_4, var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = True
    var_6 = module_0.thaw(var_4, var_5)
    var_7 = bool(var_6 == [1, [2, 3]])
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.thaw(var_4, var_5)
    var_7 = bool(var_6 == {'a': {'b': 1}})
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'y'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = {var_3: var_2}
    var_5 = module_0.m(**var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = bool(var_6 == {'a': (1, 2)})
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = 'key1'
    var_8 = bool('key1' in var_6)
    assert var_8 is True
    var_9 = 'key2'
    var_10 = bool('key2' in var_6)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = module_1.thaw(var_5)
    var_7 = bool(var_6 == {'a': None, 'b': 2})
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 2/7 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True
    var_3 = len(var_1)
    assert var_3 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = var_5['a']
    assert var_6 == 1
    var_7 = var_5['b']
    assert var_7 == 2

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = var_5['a']['b']
    assert var_6 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 2
    var_7 = var_4[2]
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1][0]
    assert var_7 == 2
    var_8 = var_5[1][1]
    assert var_8 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]['a']
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

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
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 2
    var_7 = var_4[2]
    assert var_7 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1][0]
    assert var_7 == 2
    var_8 = var_5[1][1]
    assert var_8 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = module_0.freeze(var_2)
    var_4 = var_3[0]
    assert var_4 == 1
    var_5 = var_3[var_0]
    var_6 = len(var_5)
    assert var_6 == 0

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

def test_case_0():
    var_0 = 'x'
    var_1 = 10

import pyrsistent._helpers as module_0

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
    var_13 = module_0.freeze(var_12)
    var_14 = var_13['a'][0]
    assert var_14 == 1
    var_15 = var_13['a'][1]['b']
    assert var_15 == 2
    var_16 = var_13['c'][0]
    assert var_16 == 3
    var_17 = var_13['c'][1][0]
    assert var_17 == 4
    var_18 = var_13['c'][1][1]
    assert var_18 == 5

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2, var_1]
    var_4 = set(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = 1
    var_7 = bool(1 in var_5)
    assert var_7 is True
    var_8 = 2
    var_9 = bool(2 in var_5)
    assert var_9 is True
    var_10 = 3
    var_11 = bool(3 in var_5)
    assert var_11 is True
    var_12 = len(var_5)
    assert var_12 == 3

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
    var_8 = var_7[0]
    assert var_8 == 1
    var_9 = var_7[1]['a'][0]
    assert var_9 == 2
    var_10 = var_7[1]['a'][1]
    assert var_10 == 3



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/18 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/12 statements.
# Failed to parse test_mutant_with_no_arguments.
# Partially parsed test_mutant_with_scalar_return. Retrieved 1/4 statements.


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
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 5



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 7/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = var_0[0].__name__
    var_7 = bool(var_0[0].__name__ != 'dict')
    assert var_7 is True
    var_8 = '__hash__'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 23/43 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = [var_2, var_3, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 10
    var_12 = 20
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_2.freeze(var_13)
    var_15 = [var_14]
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = [var_2, var_3]
    var_21 = module_1.pset(var_20)
    var_22 = {}
    var_23 = module_2.freeze(var_22)
    var_24 = [var_23]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/17 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 3
    var_5 = {var_3: var_4}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = {var_0, var_1, var_2}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'tags'
    var_2 = 'id'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = 2
    var_7 = 3
    var_8 = {var_3, var_6, var_7}
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = module_0.pmap()
    var_11 = [var_10]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/15 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 12/22 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 12/22 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/14 statements.
# Partially parsed test_mutant_prevents_mutation_of_input. Retrieved 5/13 statements.


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
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_0, var_1]
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 'list'
    var_9 = 'dict'
    var_10 = [var_0, var_1]
    var_11 = {var_3: var_4}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    pass

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = {var_0, var_1, var_2}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = [var_0, var_1, var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 12/29 statements.


import pyrsistent._pset as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_1.freeze(var_7)
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_1, var_10: var_2}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 6/16 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/18 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_types. Retrieved 14/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 999
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_1, var_5: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_1, var_2, var_3, var_8]

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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = module_0.pset(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'key'
    var_6 = 'original'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 99
    var_10 = [var_2, var_3, var_9]
    var_11 = 'updated'
    var_12 = {var_5: var_11}
    var_13 = module_0.pmap(var_12)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 6/16 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/24 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/24 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_deeply_nested_structure. Retrieved 12/32 statements.
# Partially parsed test_mutant_returns_scalar_value. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 999
    var_5 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 5
    var_9 = [var_1, var_2, var_3, var_8]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 'result'
    var_9 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = 3
    var_4 = module_0.pmap()
    var_5 = [var_4]
    var_6 = 15
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap(var_7)

def test_case_0():
    pass

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = 999
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = module_0.pset(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap()
    var_9 = [var_8]
    var_10 = module_0.pmap()
    var_11 = [var_10]
    var_12 = 100
    var_13 = [var_2, var_3, var_4, var_12]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = 'list'
    var_6 = 'dict'
    var_7 = module_0.pmap()
    var_8 = [var_7]
    var_9 = 'set'
    var_10 = module_1.pset()
    var_11 = [var_10]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/5 statements.
# Partially parsed test_freeze_simple_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/11 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 9/11 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 13/17 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 4/7 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 5/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/11 statements.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 9/13 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 12/17 statements.
# Partially parsed test_freeze_mixed_structure. Retrieved 18/22 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = [var_2, var_3]
    var_11 = [var_5, var_6]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_1}
    var_7 = module_1.pmap(var_6)
    var_8 = (var_7, var_3)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'data'
    var_1 = 'items'
    var_2 = 1
    var_3 = 2
    var_4 = 'nested'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = 3
    var_9 = {var_5, var_3, var_8}
    var_10 = {var_0: var_7, var_1: var_9}
    var_11 = module_0.freeze(var_10)
    var_12 = True
    var_13 = {var_4: var_12}
    var_14 = module_1.pmap(var_13)
    var_15 = [var_5, var_3, var_14]
    var_16 = [var_12, var_3, var_8]
    var_17 = module_2.pset(var_16)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/20 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_no_arguments. Retrieved 2/9 statements.
# Partially parsed test_mutant_freezes_deeply_nested_structures. Retrieved 13/27 statements.


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
    var_7 = [var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = []

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)
    var_6 = [var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = []

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = [var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_12]
    var_14 = []



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = '__hash__'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/5 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/10 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 3/7 statements.
# Partially parsed test_mutant_with_nested_dict_and_list. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_2, var_3}
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = (var_0, var_3, var_4)

def test_case_0():
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()

def test_case_0():
    var_0 = 'nested'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_original_not_modified. Retrieved 4/9 statements.


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
    var_0 = 'items'
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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    pass

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
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 9/17 statements.


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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = bool('new_key' not in var_2)
    assert var_4 is True
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 11/28 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = bool('new_key' not in var_2)
    assert var_4 is True
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True
    var_6 = 'new_key'
    var_7 = 2
    var_8 = 3
    var_9 = [var_1, var_7, var_8]
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = bool(var_9 == [1, 2, 3])
    assert var_11 is True
    var_12 = 4
    var_13 = 'x'
    var_14 = 10
    var_15 = {var_13: var_14}
    var_16 = 'test'
    var_17 = 'processed'
    var_18 = bool('processed' not in var_15)
    assert var_18 is True
    var_19 = bool(var_15 == {'x': 10})
    assert var_19 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 21/40 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = 6
    var_12 = {var_9, var_10, var_11}
    var_13 = [var_2, var_3, var_7]
    var_14 = module_1.freeze(var_13)
    var_15 = [var_14]
    var_16 = {var_9, var_10, var_11}
    var_17 = module_1.freeze(var_16)
    var_18 = [var_17]
    var_19 = 'x'
    var_20 = 10
    var_21 = {var_19: var_20}
    var_22 = {var_19: var_20}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_23]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 22/46 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = [var_2, var_3, var_8]
    var_11 = {var_2, var_3, var_8}
    var_12 = [var_2, var_3, var_8]
    var_13 = module_1.pset(var_12)
    var_14 = [var_13]
    var_15 = 'key'
    var_16 = [var_2, var_3, var_8]
    var_17 = {var_15: var_16}
    var_18 = module_0.pmap()
    var_19 = [var_18]
    var_20 = [var_2, var_3]
    var_21 = 'nested'
    var_22 = 'dict'
    var_23 = {var_21: var_22}
    var_24 = module_0.pmap()
    var_25 = [var_24]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 9/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_4: var_0}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/14 statements.
# Partially parsed test_mutant_preserves_original_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_0: var_1, var_5: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'value'
    var_4 = 3
    var_5 = {var_3: var_4}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.pmap()
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap()
    var_9 = [var_8]
    var_10 = module_0.pmap()
    var_11 = [var_10]

def test_case_0():
    pass



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_freeze_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 8/15 statements.
# Partially parsed test_freeze_set. Retrieved 5/9 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 9/13 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 5/12 statements.
# Partially parsed test_freeze_empty_dict. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_set. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 2/4 statements.
# Partially parsed test_freeze_strict_false. Retrieved 5/9 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/15 statements.
# Partially parsed test_freeze_dict_with_list_value. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_structures. Retrieved 14/27 statements.
# Partially parsed test_freeze_set_of_primitives. Retrieved 5/9 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True
    var_7 = [var_5]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': {'b': 1}})
    assert var_6 is True
    var_7 = [var_5]
    var_8 = var_5[var_0]
    var_9 = [var_8]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True
    var_7 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, [2, 3]])
    assert var_7 is True
    var_8 = [var_5]
    var_9 = var_5[var_0]
    var_10 = [var_9]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == {1, 2, 3})
    assert var_5 is True
    var_6 = [var_4]

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
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[var_0]
    var_8 = list(var_7)
    var_9 = bool(var_8 == [2, 3])
    assert var_9 is True
    var_10 = var_5[var_0]
    var_11 = [var_10]

import pyrsistent._helpers as module_0

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
    var_11 = module_0.freeze(var_10)
    var_12 = var_11['a'][0]
    assert var_12 == 1
    var_13 = var_11['a'][1]['b']
    assert var_13 == 2
    var_14 = var_11['c']
    var_15 = bool(var_11['c'] == (3, 4))
    assert var_15 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True
    var_3 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = set()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True
    var_4 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

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
    var_6 = bool(var_5 == 3.14)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.freeze(var_7)
    assert var_8 is None

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True
    var_6 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, {'a': 3}])
    assert var_7 is True
    var_8 = [var_5]
    var_9 = var_5[var_0]
    var_10 = [var_9]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = var_6[var_0]
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True
    var_10 = var_6[var_0]
    var_11 = [var_10]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = 'y'
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.freeze(var_9)
    var_11 = var_10['x'][2]['y'][0]
    assert var_11 == 3
    var_12 = [var_10]
    var_13 = var_10[var_0]
    var_14 = [var_13]
    var_15 = var_10[var_0][var_2]
    var_16 = [var_15]
    var_17 = var_10[var_0][var_2][var_3]
    var_18 = [var_17]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2, var_1, var_0}
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == {1, 2, 3})
    assert var_5 is True
    var_6 = [var_4]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 14/29 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = [var_2, var_3, var_6]
    var_8 = module_1.pset(var_7)
    var_9 = [var_2, var_3, var_6]
    var_10 = 'x'
    var_11 = 'y'
    var_12 = {var_10: var_2, var_11: var_3}
    var_13 = module_2.freeze(var_12)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 7/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 20/28 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = var_0[var_8]
    var_10 = module_0.pset(var_4)
    var_11 = var_9[0]
    var_12 = bool(var_9[0] == var_10)
    assert var_12 is True
    var_13 = module_1.pmap(var_7)
    var_14 = var_9[1]
    var_15 = bool(var_9[1] == var_13)
    assert var_15 is True
    var_16 = {}
    var_17 = module_1.pmap(var_16)
    var_18 = [var_17]
    var_19 = 'result'
    var_20 = 'map'
    var_21 = module_0.pset(var_4)
    var_22 = module_1.pmap(var_7)
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_1.pmap(var_23)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = [var_0, var_1]
    var_8 = 'x'
    var_9 = {var_8: var_0}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 9/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0[0].__name__
    assert var_5 == 'PVector'
    var_6 = 'Persistent'
    var_7 = -1
    var_8 = 'PVector'
    var_9 = -1
    var_10 = bool(var_4 == [1, 2, 3])
    assert var_10 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/9 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 12/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 9/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 11/18 statements.


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
    var_6 = 'result'
    var_7 = [var_1, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'x'
    var_7 = 5
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_0, var_1, var_3, var_4, var_10]

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 'items'
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)
    var_7 = {var_4: var_6}
    var_8 = module_1.pmap(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    pass

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'd'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0]
    var_8 = [var_2]
    var_9 = {var_4: var_5}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_function. Retrieved 1/5 statements.
# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 23/5 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'result'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = [var_7, var_8, var_9]
    var_11 = 0
    var_12 = [var_1, var_2, var_3]
    var_13 = module_0.pset(var_12)
    var_14 = [var_13]
    var_15 = [var_1, var_2, var_3]
    var_16 = {var_5: var_1}
    var_17 = module_1.pmap(var_16)
    var_18 = [var_17]
    var_19 = [var_7, var_8, var_9]
    var_20 = module_0.pset(var_19)
    var_21 = [var_20]
    var_22 = [var_7, var_8, var_9]
    var_23 = 'result'
    var_24 = {var_23: var_1}
    var_25 = module_1.pmap(var_24)
    var_26 = [var_25]

import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'result'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = [var_7, var_8, var_9]
    var_11 = 0
    var_12 = [var_1, var_2, var_3]
    var_13 = module_0.pset(var_12)
    var_14 = [var_13]
    var_15 = [var_1, var_2, var_3]
    var_16 = {var_5: var_1}
    var_17 = module_1.pmap(var_16)
    var_18 = [var_17]
    var_19 = [var_7, var_8, var_9]
    var_20 = module_0.pset(var_19)
    var_21 = [var_20]
    var_22 = [var_7, var_8, var_9]
    var_23 = 'result'
    var_24 = {var_23: var_1}
    var_25 = module_1.pmap(var_24)
    var_26 = [var_25]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_freeze_defaultdict_strict_true. Retrieved 9/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_0: var_5, var_1: var_3}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/15 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 5/8 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/8 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_empty_list. Retrieved 1/7 statements.
# Partially parsed test_mutant_with_empty_dict. Retrieved 1/7 statements.


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
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 13/41 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Plist'
    var_5 = 0
    var_6 = 'pvector'
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = 'PMap'
    var_10 = '__getitem__'
    var_11 = [var_0, var_1, var_2]
    var_12 = 4



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate_line_32. Retrieved 9/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_0: var_5, var_1: var_3}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/12 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 11/25 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/23 statements.
# Partially parsed test_mutant_freezes_set_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 8/26 statements.
# Partially parsed test_mutant_return_value_is_immutable. Retrieved 1/7 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

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
    var_9 = module_0.pmap()
    var_10 = [var_9]
    var_11 = module_0.pmap()
    var_12 = [var_11]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap()
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = module_0.pmap()
    var_8 = [var_7]

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = [var_7]

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 0
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 2
    var_8 = module_1.pset()
    var_9 = [var_8]

def test_case_0():
    var_0 = 5
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 5



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 8/23 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/23 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 10/29 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_none_return. Retrieved 3/8 statements.


import pyrsistent._pmap as module_0

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
    var_9 = bool(var_3 == [1, 2, 3])
    assert var_9 is True
    var_10 = bool(var_5 == {'a': 1})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = bool(var_3 == [{'key': 'value'}])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = [var_6]

def test_case_0():
    pass

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = bool(var_3 == {1, 2, 3})
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)
    var_8 = bool(var_7 == (1, [2, 3], {'a': 4}))
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'x'
    var_7 = {var_6: var_0}
    var_8 = bool(var_2 == [1, 2])
    assert var_8 is True
    var_9 = bool(var_5 == [3, 4])
    assert var_9 is True
    var_10 = bool(var_7 == {'x': 1})
    assert var_10 is True
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = [var_12]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = bool(var_2 == [1, 2])
    assert var_3 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 7/12 statements.


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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pmap()
    var_5 = [var_4]
    var_6 = module_1.freeze(var_3)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/18 statements.
# Failed to parse test_mutant_preserves_immutability.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 3
    var_5 = {var_3: var_4}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'value'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.pset()
    var_5 = [var_4]
    var_6 = {var_0, var_1, var_2}
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 11/18 statements.


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1
import pyrsistent._helpers as module_2

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
    var_9 = module_2.freeze(var_4)
    var_10 = module_2.freeze(var_8)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 14/34 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = '__setitem__'
    var_7 = False
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = module_1.pset(var_9)
    var_11 = 'nested'
    var_12 = [var_2, var_3, var_8]
    var_13 = {var_11: var_12}



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 3/12 statements.
# Partially parsed test_mutant_with_pmap. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_pset. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'new_key'
    var_5 = bool('new_key' not in var_2)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'x'
    var_5 = bool('x' not in var_3)
    assert var_5 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = 4
    var_7 = bool(4 not in var_4)
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'value'
    var_4 = bool(False)
    assert var_4 is True



