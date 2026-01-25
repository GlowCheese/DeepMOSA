####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_defaultdict_with_nested_values. Retrieved 4/10 statements.


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
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == [1, [2, 3]])
    assert var_6 is True

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
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 1
    var_7 = bool(1 in var_4)
    assert var_7 is True
    var_8 = 2
    var_9 = bool(2 in var_4)
    assert var_9 is True
    var_10 = 3
    var_11 = bool(3 in var_4)
    assert var_11 is True

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
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = [var_2, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = (var_9, var_10)
    var_12 = {var_0: var_8, var_1: var_11}
    var_13 = module_0.freeze(var_12)
    var_14 = bool(var_13 == {'a': [1, {'b': [2, 3]}], 'c': (4, 5)})
    assert var_14 is True

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
    var_0 = 1
    var_1 = 2
    var_2 = 'c'
    var_3 = 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = bool(var_7 == [{'a': 1}, {'b': 2}])
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': (1, 2)})
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = bool(var_4 == [1, 2])
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 'd'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.freeze(var_11)
    var_13 = bool(var_12 == {'a': {'b': {'c': [1, 2, {'d': 3}]}}})
    assert var_13 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/8 statements.
# Partially parsed test_mutant_with_list_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_preserves_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/16 statements.
# Failed to parse test_mutant_with_defaultdict_argument.


def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]

def test_case_0():
    var_0 = 'items'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 'outer'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
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
    var_0 = {}
    var_1 = []
    var_2 = set()
    var_3 = 'dict'
    var_4 = 'list'
    var_5 = 'set'

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 7/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate_line_32. Retrieved 10/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = True
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_list. Retrieved 5/9 statements.
# Partially parsed test_freeze_nested_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 6/8 statements.
# Partially parsed test_freeze_set. Retrieved 5/9 statements.
# Partially parsed test_freeze_empty_set. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_dict. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_list. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 2/4 statements.
# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_strict_false. Retrieved 5/9 statements.
# Partially parsed test_freeze_strict_true. Retrieved 5/9 statements.
# Partially parsed test_freeze_tuple_with_dict. Retrieved 6/8 statements.


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

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True
    var_6 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == [1, [2, 3]])
    assert var_6 is True
    var_7 = [var_5]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == [1, {'a': 3}])
    assert var_6 is True
    var_7 = [var_5]

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
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == {1, 2, 3})
    assert var_5 is True
    var_6 = [var_4]

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
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True
    var_3 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True
    var_3 = [var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

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
    var_12 = module_0.freeze(var_11)
    var_13 = bool(var_12 == {'a': [1, 2, {'b': 3}], 'c': (4, 5)})
    assert var_13 is True

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True
    var_6 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.freeze(var_8)
    var_10 = bool(var_9 == {'a': [{'b': 1}, {'c': 2}]})
    assert var_10 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = bool(var_7 == [(1, 2), (3, 4)])
    assert var_8 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == ({'a': 1}, 2))
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/11 statements.
# Partially parsed test_mutant_freezes_dict_argument. Retrieved 5/8 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/16 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/20 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/18 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 4/18 statements.
# Partially parsed test_mutant_deeply_nested_structure. Retrieved 9/27 statements.


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
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'x'

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
    var_6 = 0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 0
    var_3 = 1

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



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/5 statements.
# Partially parsed test_freeze_simple_list. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list. Retrieved 9/11 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 13/17 statements.
# Partially parsed test_freeze_list_with_dict_and_set. Retrieved 12/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/11 statements.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 9/13 statements.
# Partially parsed test_freeze_strict_false_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 11/19 statements.
# Partially parsed test_freeze_list_of_tuples. Retrieved 11/13 statements.
# Partially parsed test_freeze_tuple_of_lists. Retrieved 10/14 statements.


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
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_1, var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.freeze(var_5)
    var_7 = {var_0: var_1}
    var_8 = module_1.pmap(var_7)
    var_9 = [var_1, var_3]
    var_10 = module_2.pset(var_9)
    var_11 = [var_8, var_10]

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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = {var_0: var_1}
    var_6 = module_1.pmap(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = (var_1, var_2)
    var_7 = {var_0: var_6}
    var_8 = module_1.pmap(var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._helpers as module_0

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
    var_9 = module_0.freeze(var_8)
    var_10 = [var_3, var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = (var_0, var_1)
    var_9 = (var_3, var_4)
    var_10 = [var_8, var_9]

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/18 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/22 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 9/17 statements.
# Partially parsed test_mutant_preserves_immutability_of_input. Retrieved 4/9 statements.
# Failed to parse test_mutant_with_no_arguments.
# Partially parsed test_mutant_return_value_is_frozen. Retrieved 3/9 statements.


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
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = [var_5]

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

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_freeze_defaultdict_converts_to_pmap. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/19 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/18 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/16 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_empty_collections. Retrieved 6/17 statements.
# Partially parsed test_mutant_with_nested_kwargs. Retrieved 5/13 statements.


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
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 0
    var_6 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

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
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_freeze_predicate_line_1_evaluates_to_false. Retrieved 22/25 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.freeze(var_3, var_4)
    var_6 = [var_4, var_1, var_2]
    var_7 = (var_4, var_1)
    var_8 = False
    var_9 = module_0.freeze(var_7, var_8)
    var_10 = bool(var_9 == (1, 2))
    assert var_10 is True
    var_11 = [var_4, var_1, var_2]
    var_12 = set(var_11)
    var_13 = True
    var_14 = module_0.freeze(var_12, var_13)
    var_15 = [var_13, var_1, var_2]
    var_16 = module_1.pset(var_15)
    var_17 = bool(var_14 == var_16)
    assert var_17 is True
    var_18 = 42
    var_19 = True
    var_20 = module_0.freeze(var_18, var_19)
    assert var_20 == 42
    var_21 = 'hello'
    var_22 = True
    var_23 = module_0.freeze(var_21, var_22)
    assert var_23 == 'hello'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 10/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = True
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = [var_5, var_1, var_2]
    var_9 = {var_3: var_4}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_no_mutation. Retrieved 4/9 statements.
# Partially parsed test_mutant_return_value_frozen. Retrieved 1/7 statements.


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
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'original'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'y'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/12 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 7/18 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/18 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/13 statements.
# Partially parsed test_mutant_deeply_nested_structures. Retrieved 12/26 statements.


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
    var_5 = module_0.pmap()
    var_6 = [var_5]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
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
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.pmap()
    var_10 = [var_9]
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = [var_12]



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 4/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_return_value_frozen. Retrieved 2/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = module_0.pmap()
    var_5 = [var_4]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = [var_0]
    var_2 = 'data'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/19 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/19 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 5/15 statements.
# Failed to parse test_mutant_returns_frozen_value.


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
    var_8 = [var_1, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 0
    var_6 = {var_3: var_0}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'initial'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_3: var_4}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 42
    var_1 = 43
    var_2 = 44
    var_3 = [var_0, var_1, var_2]

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 0
    var_3 = {}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 5/10 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 1/12 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_freeze_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 8/15 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_set. Retrieved 7/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 9/14 statements.
# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 3/15 statements.
# Partially parsed test_freeze_empty_dict. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_set. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 2/4 statements.
# Partially parsed test_freeze_strict_false_pmap. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_true_pmap. Retrieved 9/17 statements.
# Partially parsed test_freeze_strict_false_pvector. Retrieved 5/13 statements.


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
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = [var_5]

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
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 1
    var_2 = 'a'

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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = False
    var_5 = module_1.freeze(var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True
    var_7 = [var_5]

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
    var_8 = bool(var_7 == {'a': [1, 2]})
    assert var_8 is True
    var_9 = [var_7]
    var_10 = var_7[var_0]
    var_11 = [var_10]

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
    var_2 = 'd'
    var_3 = 1
    var_4 = 2
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = (var_9, var_12)
    var_14 = 7
    var_15 = 8
    var_16 = 9
    var_17 = {var_14, var_15, var_16}
    var_18 = {var_0: var_8, var_1: var_13, var_2: var_17}
    var_19 = module_0.freeze(var_18)
    var_20 = var_19['a'][0]
    assert var_20 == 1
    var_21 = var_19['a'][2]['b']
    assert var_21 == 3
    var_22 = var_19[var_1][var_3]
    var_23 = list(var_22)
    var_24 = bool(var_23 == [5, 6])
    assert var_24 is True
    var_25 = var_19[var_2]
    var_26 = len(var_25)
    assert var_26 == 3



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/25 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 8/23 statements.


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
    var_8 = {var_3: var_0}
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
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

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

def test_case_0():
    pass

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
    var_3 = 0
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 2
    var_8 = module_1.pset()
    var_9 = [var_8]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 10/24 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/17 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 8/15 statements.
# Partially parsed test_mutant_decorator_with_nested_structure. Retrieved 9/16 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_5]
    var_7 = var_0[0]
    var_8 = [var_1, var_2, var_3]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_9]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_5]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_7]

def test_case_0():
    pass

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.freeze(var_7)
    var_9 = [var_8]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/11 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/16 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_primitive_return. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 4/18 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = 'a'
    var_6 = 'b'

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'list'
    var_3 = 'dict'



# Parsed testcases at query #27
#--------------------------




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
    var_8 = bool(var_7 is var_5)
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------




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
    var_8 = bool(var_7 is var_5)
    assert var_8 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_predicate_line_1_evaluates_to_false. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_pmap_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_pset_arguments. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = bool('new_key' not in var_2)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'initial'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'key'
    var_5 = bool('key' not in var_3)
    assert var_5 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 999
    var_6 = bool(999 not in var_4)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 'b'
    var_6 = bool('b' not in var_3)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/9 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/14 statements.
# Failed to parse test_mutant_with_set.
# Partially parsed test_mutant_with_tuple. Retrieved 2/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/15 statements.
# Failed to parse test_mutant_with_scalar_return.
# Failed to parse test_mutant_with_none_return.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 0

def test_case_0():
    pass



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 10/32 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'new_key'
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = 999
    var_10 = [var_2, var_3, var_7]
    var_11 = module_1.pset(var_10)
    var_12 = 999



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/13 statements.
# Partially parsed test_mutant_decorator_with_mutable_list. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 4/16 statements.
# Partially parsed test_mutant_decorator_multiple_args. Retrieved 6/12 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.freeze(var_3)
    var_5 = [var_4]
    var_6 = 'new_key'
    var_7 = bool('new_key' not in var_2)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = len(var_3)
    assert var_4 == 3
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {}
    var_3 = module_0.freeze(var_2)
    var_4 = [var_3]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == {'x': 1})
    assert var_6 is True
    var_7 = bool(var_5 == {'y': 2})
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 8/15 statements.
# Partially parsed test_mutant_decorator_with_mutable_input. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 12/17 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 12/17 statements.
# Partially parsed test_mutant_decorator_with_list_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/14 statements.


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
    var_8 = [var_7]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'second'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'combined'
    var_7 = {var_0: var_1}
    var_8 = {var_3: var_4}
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = module_0.freeze(var_10)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'opt'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = 'extra'
    var_8 = {var_0: var_1}
    var_9 = {var_3: var_4}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.freeze(var_10)

def test_case_0():
    pass

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.freeze(var_7)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/20 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/18 statements.
# Partially parsed test_mutant_preserves_immutability. Retrieved 2/7 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.


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
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 'list'
    var_8 = {var_3: var_0}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 'a'
    var_8 = {var_3: var_0}
    var_9 = module_0.pmap(var_8)

def test_case_0():
    var_0 = 42
    var_1 = [var_0]

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

def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_freeze_defaultdict_converts_to_pmap. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_freeze_dict_basic. Retrieved 8/10 statements.
# Partially parsed test_freeze_dict_nested. Retrieved 11/15 statements.
# Partially parsed test_freeze_list_basic. Retrieved 6/9 statements.
# Partially parsed test_freeze_list_nested. Retrieved 10/15 statements.
# Partially parsed test_freeze_set. Retrieved 8/10 statements.
# Partially parsed test_freeze_tuple_basic. Retrieved 5/6 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 8/12 statements.
# Partially parsed test_freeze_mixed_nested. Retrieved 17/24 statements.
# Partially parsed test_freeze_empty_dict. Retrieved 5/7 statements.
# Partially parsed test_freeze_empty_list. Retrieved 4/7 statements.
# Partially parsed test_freeze_empty_set. Retrieved 5/7 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 3/4 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/17 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 8/10 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_deeply_nested_structure. Retrieved 29/43 statements.
# Partially parsed test_freeze_set_with_multiple_elements. Retrieved 10/13 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_7]
    var_9 = var_5['a']
    assert var_9 == 1
    var_10 = var_5['b']
    assert var_10 == 2

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_7]
    var_9 = var_5[var_0]
    var_10 = {}
    var_11 = module_1.pmap(var_10)
    var_12 = [var_11]
    var_13 = var_5['a']['b']
    assert var_13 == 3

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = []
    var_6 = var_4[0]
    assert var_6 == 1
    var_7 = var_4[1]
    assert var_7 == 2
    var_8 = var_4[2]
    assert var_8 == 3

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = []
    var_7 = var_5[0]
    assert var_7 == 1
    var_8 = var_5[var_0]
    var_9 = {}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_10]
    var_12 = var_5[1]['a']
    assert var_12 == 3

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = []
    var_7 = module_1.pset(var_6)
    var_8 = [var_7]
    var_9 = 1
    var_10 = bool(1 in var_5)
    assert var_10 is True
    var_11 = 2
    var_12 = bool(2 in var_5)
    assert var_12 is True
    var_13 = 3
    var_14 = bool(3 in var_5)
    assert var_14 is True

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
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[var_0]
    var_8 = []
    var_9 = var_5[1][0]
    assert var_9 == 2

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

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
    var_9 = module_0.freeze(var_8)
    var_10 = {}
    var_11 = module_1.pmap(var_10)
    var_12 = [var_11]
    var_13 = var_9[var_0]
    var_14 = []
    var_15 = var_9[var_1]
    var_16 = {}
    var_17 = module_1.pmap(var_16)
    var_18 = [var_17]
    var_19 = var_9['dict']['nested']
    assert var_19 == 'value'

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = [var_3]
    var_5 = len(var_1)
    assert var_5 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = len(var_1)
    assert var_3 == 0

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = module_1.pset(var_2)
    var_4 = [var_3]
    var_5 = len(var_1)
    assert var_5 == 0

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

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
    var_7 = True
    var_8 = module_0.freeze(var_7)
    assert var_8 is True
    var_9 = None
    var_10 = module_0.freeze(var_9)
    assert var_10 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = []

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

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
    var_9 = var_5['a']
    assert var_9 == 1

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
    var_4 = True
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = 7
    var_13 = [var_11, var_12]
    var_14 = (var_10, var_13)
    var_15 = {var_0: var_9, var_1: var_14}
    var_16 = module_0.freeze(var_15)
    var_17 = {}
    var_18 = module_1.pmap(var_17)
    var_19 = [var_18]
    var_20 = var_16[var_0]
    var_21 = []
    var_22 = var_16[var_0][var_2]
    var_23 = {}
    var_24 = module_1.pmap(var_23)
    var_25 = [var_24]
    var_26 = var_16[var_0][var_2][var_3]
    var_27 = []
    var_28 = var_16[var_1]
    var_29 = var_16[var_1][var_2]
    var_30 = []

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = {var_0, var_1, var_2, var_3, var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = []
    var_8 = module_1.pset(var_7)
    var_9 = [var_8]
    var_10 = len(var_6)
    assert var_10 == 5



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_decorator_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate_line_32. Retrieved 9/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = True
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = {var_3: var_4}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 7/17 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 14/19 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 14/19 statements.
# Partially parsed test_mutant_decorator_with_pmap. Retrieved 8/13 statements.
# Partially parsed test_mutant_decorator_with_pset. Retrieved 7/12 statements.


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_5]
    var_7 = '__hash__'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'combined'
    var_7 = {var_0: var_1}
    var_8 = module_0.freeze(var_7)
    var_9 = {var_3: var_4}
    var_10 = module_0.freeze(var_9)
    var_11 = [var_8, var_10]
    var_12 = {var_6: var_11}
    var_13 = module_0.freeze(var_12)

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'info'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = 'extra'
    var_8 = {var_0: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_3: var_4}
    var_11 = module_0.freeze(var_10)
    var_12 = {var_6: var_9, var_7: var_11}
    var_13 = module_0.freeze(var_12)

def test_case_0():
    pass

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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_nested_structure. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_scalar_return. Retrieved 4/7 statements.
# Partially parsed test_mutant_with_string_argument. Retrieved 1/4 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_function. Retrieved 1/5 statements.
# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 11/5 statements.


def test_case_0():
    var_0 = 'result'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'b'
    var_8 = {var_7: var_2}
    var_9 = 0
    var_10 = '__hash__'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'result'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'b'
    var_8 = {var_7: var_2}
    var_9 = 0
    var_10 = '__hash__'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 12/25 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = module_0.pmap()
    var_8 = [var_7]
    var_9 = 'input_dict'
    var_10 = module_0.pmap()
    var_11 = [var_10]
    var_12 = 'input_set'
    var_13 = module_1.pset()
    var_14 = [var_13]
    var_15 = bool(var_2 == {'key': 'value'})
    assert var_15 is True
    var_16 = bool(var_6 == {1, 2, 3})
    assert var_16 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_freeze_defaultdict_predicate. Retrieved 8/13 statements.


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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 10/21 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_9 = module_1.pset(var_8)
    var_10 = [var_2, var_3, var_7]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/11 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_freezes_set_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_freezes_tuple_arguments. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_empty_collections. Retrieved 6/18 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 26/36 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]

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
    var_3 = 'nested'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

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
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]

def test_case_0():
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'tuple'
    var_3 = 'set'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = [var_4, var_5, var_8]
    var_10 = 'nested'
    var_11 = 'deep'
    var_12 = 5
    var_13 = 6
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_10: var_15}
    var_17 = 7
    var_18 = 8
    var_19 = 9
    var_20 = [var_18, var_19]
    var_21 = (var_17, var_20)
    var_22 = 10
    var_23 = 11
    var_24 = {var_22, var_23}
    var_25 = {var_0: var_9, var_1: var_16, var_2: var_21, var_3: var_24}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_list_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/16 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_primitive_arguments. Retrieved 2/5 statements.


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
    var_5 = [var_0, var_1, var_2, var_4]

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}

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
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = 'c'

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
    var_0 = {}
    var_1 = []
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 5
    var_1 = 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_freeze_dict. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_nested_list. Retrieved 8/15 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 7/14 statements.
# Partially parsed test_freeze_set. Retrieved 5/9 statements.
# Partially parsed test_freeze_tuple. Retrieved 5/7 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 9/14 statements.
# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_strict_true. Retrieved 4/12 statements.
# Partially parsed test_freeze_strict_false. Retrieved 4/12 statements.


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
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = 3.14
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == 3.14)
    assert var_6 is True

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = {}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {})
    assert var_6 is True
    var_7 = set()
    var_8 = module_0.freeze(var_7)
    var_9 = set()
    var_10 = bool(var_8 == var_9)
    assert var_10 is True
    var_11 = ()
    var_12 = module_0.freeze(var_11)
    var_13 = bool(var_12 == ())
    assert var_13 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'tuple'
    var_3 = 'set'
    var_4 = 1
    var_5 = 2
    var_6 = 'nested'
    var_7 = {var_6: var_1}
    var_8 = [var_4, var_5, var_7]
    var_9 = 'inner'
    var_10 = 3
    var_11 = 4
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = 5
    var_15 = 6
    var_16 = 7
    var_17 = [var_15, var_16]
    var_18 = (var_14, var_17)
    var_19 = 8
    var_20 = 9
    var_21 = {var_19, var_20}
    var_22 = {var_0: var_8, var_1: var_13, var_2: var_18, var_3: var_21}
    var_23 = module_0.freeze(var_22)
    var_24 = var_23['list'][2]
    var_25 = bool(var_23['list'][2] == {'nested': 'dict'})
    assert var_25 is True
    var_26 = var_23['dict']['inner']
    var_27 = bool(var_23['dict']['inner'] == [3, 4])
    assert var_27 is True
    var_28 = var_23['tuple'][1]
    var_29 = bool(var_23['tuple'][1] == [6, 7])
    assert var_29 is True
    var_30 = var_23['set']
    var_31 = bool(var_23['set'] == {8, 9})
    assert var_31 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/13 statements.
# Partially parsed test_mutant_with_nested_structure. Retrieved 7/18 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_scalar_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_kwargs_and_args. Retrieved 8/18 statements.


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
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2, var_3]

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

def test_case_0():
    pass

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 'list'
    var_7 = 'dict'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_thaw_pvector_to_list. Retrieved 3/7 statements.
# Partially parsed test_thaw_pmap_to_dict. Retrieved 4/6 statements.
# Partially parsed test_thaw_pset_to_set. Retrieved 3/7 statements.
# Partially parsed test_thaw_nested_pvector_and_pmap. Retrieved 3/9 statements.
# Partially parsed test_thaw_tuple_recursive. Retrieved 3/10 statements.
# Failed to parse test_thaw_empty_pvector.
# Partially parsed test_thaw_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_thaw_empty_pset. Retrieved 1/5 statements.
# Partially parsed test_thaw_list_strict_true. Retrieved 6/7 statements.
# Partially parsed test_thaw_dict_strict_true. Retrieved 7/8 statements.
# Partially parsed test_thaw_nested_list_strict_true. Retrieved 7/9 statements.
# Partially parsed test_thaw_nested_dict_strict_true. Retrieved 4/10 statements.
# Partially parsed test_thaw_list_strict_false. Retrieved 6/7 statements.
# Partially parsed test_thaw_dict_strict_false. Retrieved 7/8 statements.
# Partially parsed test_thaw_deeply_nested_structures. Retrieved 7/20 statements.
# Partially parsed test_thaw_tuple_with_primitives. Retrieved 5/6 statements.
# Partially parsed test_thaw_pmap_with_pmap_values. Retrieved 6/9 statements.


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
    var_2 = 3
    var_3 = [var_1, var_2]

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
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'hello'
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

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]
    var_6 = True
    var_7 = module_1.thaw(var_5, var_6)
    var_8 = bool(var_7 == [1, {'x': 2}])
    assert var_8 is True
    var_9 = var_7[var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = True

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 0
    var_7 = 'a'
    var_8 = 'b'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 3.14
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.thaw(var_3)
    var_5 = bool(var_4 == (1, 'hello', 3.14))
    assert var_5 is True

import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 5
    var_1 = 'inner'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'outer'
    var_5 = {var_4: var_3}
    var_6 = module_0.m(**var_5)
    var_7 = module_1.thaw(var_6)
    var_8 = bool(var_7 == {'outer': {'inner': 5}})
    assert var_8 is True
    var_9 = 'outer'
    var_10 = var_7[var_9]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_freeze_list_to_pvector. Retrieved 6/8 statements.
# Partially parsed test_freeze_nested_list. Retrieved 7/11 statements.
# Partially parsed test_freeze_tuple_recursive. Retrieved 7/10 statements.
# Partially parsed test_freeze_complex_nested_structure. Retrieved 13/17 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 9/11 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/5 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 7/11 statements.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 9/13 statements.
# Partially parsed test_freeze_tuple_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_freeze_strict_false_pvector. Retrieved 5/9 statements.
# Partially parsed test_freeze_list_with_set. Retrieved 9/11 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_1.pset(var_7)
    var_9 = {var_0: var_8}
    var_10 = module_2.pmap(var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_1.pset(var_6)
    var_8 = [var_7]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 18/45 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_12 = 10
    var_13 = 20
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 5
    var_16 = {var_0: var_15}
    var_17 = module_0.pmap(var_16)



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

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/11 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 7/13 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/8 statements.
# Partially parsed test_mutant_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/18 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 6/14 statements.


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
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.pmap(var_5)

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
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_dict_basic. Retrieved 6/10 statements.
# Partially parsed test_freeze_dict_nested. Retrieved 7/14 statements.
# Partially parsed test_freeze_list_basic. Retrieved 6/10 statements.
# Partially parsed test_freeze_list_nested. Retrieved 8/15 statements.
# Partially parsed test_freeze_tuple_basic. Retrieved 5/7 statements.
# Partially parsed test_freeze_tuple_nested. Retrieved 7/11 statements.
# Partially parsed test_freeze_set_basic. Retrieved 5/9 statements.
# Partially parsed test_freeze_set_from_list. Retrieved 5/9 statements.
# Partially parsed test_freeze_empty_dict. Retrieved 2/6 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/7 statements.
# Partially parsed test_freeze_empty_tuple. Retrieved 2/4 statements.
# Failed to parse test_freeze_defaultdict.
# Partially parsed test_freeze_defaultdict_nested. Retrieved 3/12 statements.
# Partially parsed test_freeze_deeply_nested. Retrieved 15/29 statements.
# Partially parsed test_freeze_strict_false_dict. Retrieved 5/9 statements.
# Partially parsed test_freeze_strict_false_list. Retrieved 6/10 statements.
# Partially parsed test_freeze_dict_with_list_value. Retrieved 8/12 statements.
# Partially parsed test_freeze_list_with_dict_and_tuple. Retrieved 11/16 statements.


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
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = bool(var_5 == {'a': {'b': 3}})
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
    var_8 = [var_7]

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
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == {1, 2})
    assert var_5 is True
    var_6 = [var_4]

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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = {var_2: var_6}
    var_8 = [var_1, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.freeze(var_9)
    var_11 = var_10['a'][0]
    assert var_11 == 1
    var_12 = [var_10]
    var_13 = var_10[var_0]
    var_14 = [var_13]
    var_15 = var_10[var_0][var_1]
    var_16 = [var_15]
    var_17 = var_10[var_0][var_1][var_2]
    var_18 = var_10[var_0][var_1][var_2][var_1]
    var_19 = [var_18]

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
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.freeze(var_2, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True
    var_7 = [var_4]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.freeze(var_5)
    var_7 = bool(var_6 == {'key': [1, 2, 3]})
    assert var_7 is True
    var_8 = var_6[var_0]
    var_9 = [var_8]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.freeze(var_6)
    var_8 = var_7[0]
    var_9 = bool(var_7[0] == {'a': 1})
    assert var_9 is True
    var_10 = var_7[1]
    var_11 = bool(var_7[1] == (2, 3))
    assert var_11 is True
    var_12 = 0
    var_13 = var_7[var_12]
    var_14 = [var_13]
    var_15 = var_7[var_1]

import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'values'
    var_2 = 'config'
    var_3 = 'test'
    var_4 = 1
    var_5 = 2
    var_6 = 'nested'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 'debug'
    var_11 = 'items'
    var_12 = False
    var_13 = 4
    var_14 = 5
    var_15 = 6
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_10: var_12, var_11: var_16}
    var_18 = {var_0: var_3, var_1: var_9, var_2: var_17}
    var_19 = module_0.freeze(var_18)
    var_20 = var_19['name']
    assert var_20 == 'test'
    var_21 = var_19['values'][2]['nested']
    assert var_21 is True
    var_22 = var_19['config']['items'][0]
    assert var_22 == 4



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 10/21 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 11/26 statements.
# Partially parsed test_mutant_with_deeply_nested_dict. Retrieved 13/26 statements.
# Failed to parse test_mutant_returns_frozen_result.


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
    var_5 = {var_0: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = []
    var_8 = {}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_9]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

def test_case_0():
    pass

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
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 0
    var_8 = []
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_10]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.pmap(var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_14]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 11/25 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/23 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 4/11 statements.
# Partially parsed test_mutant_prevents_mutation_of_input. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 8/18 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 8/23 statements.


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
    var_9 = module_0.pmap()
    var_10 = [var_9]
    var_11 = module_0.pmap()
    var_12 = [var_11]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = [var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap()
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = list(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

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
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    pass

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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 8/12 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 17/32 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = 4
    var_15 = 5
    var_16 = [var_14, var_15]
    var_17 = module_1.pset(var_16)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 13/31 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = '__hash__'
    var_11 = 'x'
    var_12 = 'y'
    var_13 = {var_11: var_2, var_12: var_3}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_14]



# Parsed testcases at query #14
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
    var_6 = {var_0: var_5, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.pmap()
    var_9 = [var_8]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 4/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 10/25 statements.
# Partially parsed test_mutant_with_deeply_nested_structures. Retrieved 11/18 statements.


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
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_0, var_5]
    var_7 = [var_0]
    var_8 = {var_1: var_0}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_9]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = {var_3: var_0}
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]

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

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0]
    var_6 = 0
    var_7 = [var_0]
    var_8 = {var_3: var_0}
    var_9 = module_0.pmap(var_8)
    var_10 = [var_9]

import pyrsistent._pmap as module_0

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
    var_9 = {var_0: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_10]



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/15 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 8/19 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 8/22 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 15/22 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Failed to parse test_mutant_return_value_is_frozen.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 9/22 statements.


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
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1]
    var_7 = [var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_0: var_1}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_3: var_4}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_8: var_11, var_9: var_13}
    var_15 = module_0.pmap(var_14)

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
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.pmap()
    var_8 = [var_7]
    var_9 = [var_2, var_3]



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_structure. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 5/11 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_return_value_is_frozen. Retrieved 3/9 statements.
# Partially parsed test_mutant_with_nested_list_in_dict. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

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
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 'nested'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/15 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 6/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/15 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/20 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 9/16 statements.
# Partially parsed test_mutant_original_arguments_unchanged. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_empty_collections. Retrieved 8/14 statements.


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
    var_3 = module_0.pmap()
    var_4 = [var_3]
    var_5 = {var_0: var_1}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'nested'
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
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_4}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = module_0.pmap()
    var_5 = [var_4]
    var_6 = {var_0: var_1}
    var_7 = module_0.pmap(var_6)

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
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_3, var_6)
    var_8 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = var_3[0]
    assert var_5 == 1

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
    var_6 = set()
    var_7 = module_1.pset(var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 6/11 statements.
# Partially parsed test_mutant_nested_structures. Retrieved 12/20 statements.
# Partially parsed test_mutant_preserves_original_argument. Retrieved 4/11 statements.
# Partially parsed test_mutant_multiple_arguments. Retrieved 10/18 statements.
# Partially parsed test_mutant_with_set. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 5/10 statements.


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
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap(var_4)

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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 'list'
    var_6 = 'dict'
    var_7 = [var_0, var_1]
    var_8 = {var_3: var_0}
    var_9 = module_0.pmap(var_8)

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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '__hash__'
    var_6 = var_0[0].__name__
    var_7 = bool(var_0[0].__name__ in ('pvector', 'pset', 'pmap'))
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 9/21 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

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
    var_9 = [var_5]
    var_10 = [var_8]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 5/13 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 9/20 statements.
# Partially parsed test_mutant_freezes_return_value. Retrieved 1/9 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/15 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_scalar_values. Retrieved 1/5 statements.
# Partially parsed test_mutant_deeply_nested_structure. Retrieved 15/33 statements.


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
    var_5 = {var_0: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = []

def test_case_0():
    var_0 = []

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 0
    var_6 = []
    var_7 = {}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]

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
    var_5 = []

def test_case_0():
    pass

def test_case_0():
    var_0 = 42

import pyrsistent._pmap as module_0

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
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_14]
    var_16 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = True
    assert var_4 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 2/7 statements.
# Partially parsed test_mutant_nested_structure. Retrieved 9/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/9 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 3/8 statements.
# Failed to parse test_mutant_return_value_is_frozen.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 5
    var_1 = 20

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
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 10/24 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = module_1.pset(var_8)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
    assert var_4 is True
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = bool(var_7 == {'a': 1, 'b': 2})
    assert var_8 is True
    var_9 = 10
    var_10 = 20



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 8/12 statements.


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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 11/27 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 10/24 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 7/17 statements.
# Partially parsed test_mutant_with_sets. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_tuple. Retrieved 7/19 statements.
# Partially parsed test_mutant_multiple_args. Retrieved 11/28 statements.


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
    var_9 = [var_0, var_1, var_2]
    var_10 = {var_4: var_0}
    var_11 = module_0.pmap(var_10)

def test_case_0():
    pass

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 'inner'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap()
    var_9 = [var_8]
    var_10 = module_0.pmap()
    var_11 = [var_10]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'nested'
    var_2 = 'dict'
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap()
    var_5 = [var_4]
    var_6 = 'b'
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
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)

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
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 5
    var_7 = 6
    var_8 = (var_6, var_7)
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = [var_10]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1. Retrieved 9/21 statements.


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
    var_9 = [var_4]
    var_10 = [var_8]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 5/15 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 6/14 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/19 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 9/20 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 9/18 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 7/15 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_scalar_return. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 10/21 statements.


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
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap()
    var_7 = [var_6]
    var_8 = [var_1, var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1, var_2]
    var_7 = {var_4: var_0}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'original'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
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
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    pass

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
    var_4 = {var_3: var_0}
    var_5 = {var_0, var_1}
    var_6 = [var_0, var_1]
    var_7 = {var_3: var_0}
    var_8 = module_0.pmap(var_7)
    var_9 = 'set'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_multiple_arguments. Retrieved 6/10 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_with_pset. Retrieved 5/10 statements.
# Partially parsed test_mutant_decorator_with_pmap. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    pass

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 7/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0[0].__name__
    assert var_5 == 'PVector'
    var_6 = '__hash__'
    var_7 = "<class 'pyrsistent._pvector.PVector'>"



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 3/8 statements.
# Partially parsed test_mutant_freezes_list_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/14 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 2/9 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 11/25 statements.
# Failed to parse test_mutant_returns_frozen_primitive.
# Partially parsed test_mutant_with_empty_containers. Retrieved 6/16 statements.


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
    pass

def test_case_0():
    var_0 = 'inner'
    var_1 = 'value'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'merged'

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'

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
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = 'nested'

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = set()
    var_3 = 'dict'
    var_4 = 'list'
    var_5 = 'set'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 10/18 statements.


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
    var_9 = len(var_0)
    assert var_9 == 1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = True
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/13 statements.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 11/25 statements.
# Partially parsed test_mutant_freezes_dict_argument. Retrieved 6/13 statements.
# Partially parsed test_mutant_freezes_set_argument. Retrieved 5/12 statements.
# Partially parsed test_mutant_freezes_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_freezes_kwargs. Retrieved 8/18 statements.
# Partially parsed test_mutant_multiple_arguments. Retrieved 8/23 statements.
# Partially parsed test_mutant_preserves_function_behavior. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_defaultdict. Retrieved 4/13 statements.
# Partially parsed test_mutant_with_complex_nested_structure. Retrieved 22/38 statements.


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
    var_9 = module_0.pmap()
    var_10 = [var_9]
    var_11 = module_0.pmap()
    var_12 = [var_11]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]

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

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = 'a'
    var_8 = module_0.pmap()
    var_9 = [var_8]

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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'tuple'
    var_3 = 1
    var_4 = 2
    var_5 = 'nested'
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 5
    var_12 = 6
    var_13 = {var_11, var_12}
    var_14 = 7
    var_15 = 8
    var_16 = 9
    var_17 = [var_15, var_16]
    var_18 = (var_14, var_17)
    var_19 = {var_0: var_10, var_1: var_13, var_2: var_18}
    var_20 = module_0.pmap()
    var_21 = [var_20]
    var_22 = module_1.pset()
    var_23 = [var_22]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return_value. Retrieved 14/26 statements.


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = '__class__'
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = module_1.pset(var_8)
    var_10 = 'x'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = module_0.pmap(var_12)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_mutant_decorator_predicate_line_1_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'initial'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2 == {'initial': 'data'})
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_mutant_freezes_arguments_and_return_value. Retrieved 4/11 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 3/6 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 6/13 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 2/9 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 6/17 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_empty_containers. Retrieved 3/10 statements.
# Partially parsed test_mutant_with_deeply_nested_structure. Retrieved 10/28 statements.


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
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 3
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 10
    var_1 = 20

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
    pass

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()

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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 5/11 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_with_nested_structures. Retrieved 8/22 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/22 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/13 statements.
# Partially parsed test_mutant_with_scalar_values. Retrieved 1/4 statements.
# Partially parsed test_mutant_with_mixed_args_and_kwargs. Retrieved 11/29 statements.


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
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 'nested'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 4
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_mutant_freezes_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_freezes_dict_arguments. Retrieved 5/11 statements.
# Failed to parse test_mutant_freezes_return_value.
# Partially parsed test_mutant_freezes_nested_structures. Retrieved 6/16 statements.
# Partially parsed test_mutant_with_multiple_arguments. Retrieved 7/21 statements.
# Partially parsed test_mutant_with_kwargs. Retrieved 8/14 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/13 statements.
# Failed to parse test_mutant_with_primitive_return.
# Failed to parse test_mutant_with_none_return.


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
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'nested'
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

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



