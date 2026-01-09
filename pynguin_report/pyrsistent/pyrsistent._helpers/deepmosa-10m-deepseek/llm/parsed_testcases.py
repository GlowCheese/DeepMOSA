####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_mutant_with_positional_arguments. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 11/18 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 4/11 statements.
# Failed to parse test_mutant_preserves_function_metadata.
# Partially parsed test_mutant_with_no_arguments. Retrieved 4/9 statements.
# Partially parsed test_mutant_with_nested_mutables. Retrieved 11/16 statements.
# Partially parsed test_mutant_freezes_arguments_recursively. Retrieved 8/13 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = bool(var_1 == [1])
    assert var_2 is True
    var_3 = 2
    var_4 = [var_3]

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = bool(var_2 == {'a': 1})
    assert var_4 is True
    var_5 = {var_0: var_3}
    var_6 = module_0.pmap(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'initial'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = bool(var_2 == [1, 2])
    assert var_7 is True
    var_8 = bool(var_5 == {'initial': 0})
    assert var_8 is True
    var_9 = [var_0, var_1, var_6]
    var_10 = 'factor'
    var_11 = {var_3: var_4, var_10: var_6}
    var_12 = module_0.pmap(var_11)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 'inner'
    var_1 = 'new'
    var_2 = 'old'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = bool(var_6 == {'inner': {'new': 1, 'old': 2}})
    assert var_7 is True
    var_8 = {var_2: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_9}
    var_11 = module_0.pmap(var_10)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = bool(var_6 == [[1, 2], [3, 4]])
    assert var_7 is True
    var_8 = [var_0, var_1]

import pyrsistent._pset as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True
    var_5 = {var_0, var_1, var_3}
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = bool(var_4 == (1, [2, 3]))
    assert var_5 is True
    var_6 = [var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 9/19 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_freezes_defaultdict. Retrieved 4/12 statements.
# Partially parsed test_mutant_decorator_with_multiple_args. Retrieved 10/24 statements.
# Partially parsed test_mutant_decorator_returns_non_container_unchanged. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_freezes_empty_structures. Retrieved 3/14 statements.


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
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = bool(var_8 == {'list': [1, 2], 'tuple': (3, 4)})
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = bool(var_3 == {1, 2, 3})
    assert var_5 is True

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
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = bool(var_2 == [1, 2])
    assert var_9 is True
    var_10 = bool(var_5 == {'x': 10})
    assert var_10 is True
    var_11 = bool(var_8 == {3, 4})
    assert var_11 is True
    var_12 = 0

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 2/3 statements.
# Partially parsed test_freeze_list_with_ints. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/9 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_dict_with_list_values. Retrieved 7/10 statements.
# Partially parsed test_freeze_tuple_with_list. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 3/10 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.


import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]

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


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]


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

import pyrsistent._pset as module_1


def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True


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
    var_3 = [var_2]
    var_4 = (var_1, var_3)
    var_5 = (var_0, var_4)
    var_6 = module_0.freeze(var_5)
    var_7 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/24 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 12/23 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 7/13 statements.
# Partially parsed test_mutant_decorator_strict_false_behavior. Retrieved 7/12 statements.



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
    var_12 = 'value'
    var_13 = {var_4: var_0, var_11: var_12}
    var_14 = module_0.pmap(var_13)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 3
    var_1 = 4

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

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 100
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 12/15 statements.
# Partially parsed test_freeze_tuple. Retrieved 11/13 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 9/14 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false. Retrieved 10/11 statements.
# Partially parsed test_freeze_strict_true. Retrieved 10/11 statements.


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 'y'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = [var_4]
    var_6 = (var_0, var_3, var_5)
    var_7 = module_0.freeze(var_6)
    var_8 = {var_1: var_2}
    var_9 = module_1.pmap(var_8)
    var_10 = [var_4]

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.freeze(var_6)
    var_8 = [var_2, var_3]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = module_1.freeze(var_4, var_5)
    var_7 = {var_0: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = [var_3]
    var_5 = True
    var_6 = module_1.freeze(var_4, var_5)
    var_7 = {var_0: var_5}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_8]

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = None
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_mutant_with_list_argument. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_dict_argument. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_set_argument. Retrieved 6/11 statements.
# Partially parsed test_mutant_with_tuple_argument. Retrieved 5/10 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 8/13 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 4/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 14/22 statements.
# Partially parsed test_mutant_with_no_arguments. Retrieved 4/8 statements.
# Failed to parse test_mutant_preserves_function_metadata.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True
    var_6 = [var_0, var_1, var_2, var_4]

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pset as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = bool(var_2 == {1, 2})
    assert var_4 is True
    var_5 = [var_0, var_1, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = bool(var_3 == (1, 2, 3))
    assert var_4 is True
    var_5 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_1, var_3: var_4}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pset as module_0


def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = bool(var_9 == {'list': [1, 2, 3], 'set': {4, 5}})
    assert var_10 is True
    var_11 = 99
    var_12 = [var_11, var_3, var_4]
    var_13 = [var_6, var_7]
    var_14 = module_0.pset(var_13)

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'answer'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 8/9 statements.
# Partially parsed test_freeze_tuple_with_elements. Retrieved 11/13 statements.
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


def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_1.pmap(var_5)
    var_7 = [var_6]

import pyrsistent._pset as module_1


def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = module_1.pset()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = ()
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_freeze_pmap_with_strict_true. Retrieved 15/19 statements.
# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 13/20 statements.



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
    var_9 = True
    var_10 = [var_9, var_3]
    var_11 = {var_5: var_6}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict_argument. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set_argument. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple_argument. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 2/7 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/20 statements.
# Failed to parse test_mutant_decorator_with_no_arguments.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 4/13 statements.
# Failed to parse test_mutant_decorator_freezes_returned_set.


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
    var_0 = 10
    var_1 = 20

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
    var_0 = 1
    var_1 = 2
    var_2 = 0
    var_3 = [var_2]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_with_list_arg. Retrieved 9/10 statements.
# Partially parsed test_mutant_with_nested_structures. Retrieved 17/20 statements.
# Partially parsed test_mutant_preserves_input_immutability. Retrieved 15/17 statements.
# Partially parsed test_mutant_with_defaultdict. Retrieved 9/15 statements.
# Partially parsed test_mutant_with_pvector_arg. Retrieved 8/11 statements.
# Partially parsed test_mutant_with_function_modifying_multiple_args. Retrieved 12/14 statements.


import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = var_1(var_2, var_3)
    assert var_4 == 3


def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = module_0.mutant(var_0)
    var_2 = 3
    var_3 = 4
    var_4 = var_1(a=var_2, b=var_3)
    assert var_4 == 12


def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = var_1(var_2, y=var_3, z=var_4)
    assert var_5 == 6


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

import pyrsistent._pset as module_1


def test_case_0():
    var_0 = 4
    var_1 = lambda s: s.add(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = var_2(var_6)
    var_8 = [var_3, var_4, var_5, var_0]
    var_9 = module_1.pset(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = 'list'
    var_1 = 4
    var_2 = lambda data: data[var_0].append(var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = 'dict'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_8, var_4: var_11}
    var_13 = var_3(var_12)
    var_14 = [var_5, var_6, var_7, var_1]
    var_15 = {var_9: var_10}
    var_16 = module_1.pmap(var_15)


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


def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0
    var_2 = module_0.mutant(var_1)
    var_3 = var_2()
    assert var_3 == 42


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 4
    var_7 = 'b'
    var_8 = {var_7: var_1}
    var_9 = lambda lst, dct: (lst.append(var_6), dct.update(var_8))
    var_10 = module_0.mutant(var_9)
    var_11 = var_10(var_3, var_5)
    var_12 = bool(var_3 == [1, 2, 3])
    assert var_12 is True
    var_13 = bool(var_5 == {'a': 1})
    assert var_13 is True
    var_14 = [var_0, var_1, var_2, var_6]
    var_15 = {var_4: var_0, var_7: var_1}
    var_16 = module_1.pmap(var_15)


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = lambda d: d[var_0].append(var_5)
    var_7 = module_0.mutant(var_6)
    var_8 = [var_1, var_2, var_5]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = lambda v: v.append(var_4)
    var_6 = module_0.mutant(var_5)
    var_7 = [var_0, var_1, var_2, var_4]

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'z'
    var_7 = 3
    var_8 = lambda m: m.set(var_6, var_7)
    var_9 = module_1.mutant(var_8)
    var_10 = var_9(var_5)
    var_11 = {var_0: var_2, var_1: var_3, var_6: var_7}
    var_12 = module_0.pmap(var_11)
    var_13 = bool(var_10 == var_12)
    assert var_13 is True

import pyrsistent._pset as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = 4
    var_6 = lambda s: s.add(var_5)
    var_7 = module_1.mutant(var_6)
    var_8 = var_7(var_4)
    var_9 = [var_0, var_1, var_2, var_5]
    var_10 = module_0.pset(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = module_0.mutant(var_1)
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 is None

import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = 1
    var_1 = 'k'
    var_2 = 'v'
    var_3 = {var_1: var_2}
    var_4 = lambda a, b: (a.append(var_0), b.update(var_3))
    var_5 = module_0.mutant(var_4)
    var_6 = []
    var_7 = {}
    var_8 = var_5(var_6, var_7)
    var_9 = [var_0]
    var_10 = {var_1: var_2}
    var_11 = module_1.pmap(var_10)


def test_case_0():
    var_0 = lambda *args: sum(args)
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = var_1(var_2, var_3, var_4, var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = lambda **kwargs: sum(kwargs.values())
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = var_1(a=var_2, b=var_3, c=var_4)
    assert var_5 == 6



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_freeze_empty_list. Retrieved 3/4 statements.
# Partially parsed test_freeze_list_with_elements. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 10/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 3/10 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 4/6 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 5/8 statements.
# Partially parsed test_freeze_nested_mixed. Retrieved 16/19 statements.



def test_case_0():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    var_2 = []


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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

import pyrsistent._pset as module_1


def test_case_0():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = module_1.pset(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = [var_3, var_1]


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

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    assert var_1 == 42


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    assert var_1 == 'hello'

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
    var_13 = [var_6, var_7]
    var_14 = module_1.pset(var_13)
    var_15 = (var_5, var_14)



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 10/11 statements.
# Partially parsed test_freeze_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_freeze_with_strict_false. Retrieved 11/12 statements.
# Partially parsed test_freeze_with_strict_true. Retrieved 10/15 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.


import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_thaw_pvector. Retrieved 3/7 statements.
# Partially parsed test_thaw_pvector_recursive. Retrieved 3/9 statements.
# Partially parsed test_thaw_pmap. Retrieved 4/6 statements.
# Partially parsed test_thaw_pmap_recursive. Retrieved 3/10 statements.
# Partially parsed test_thaw_pset. Retrieved 3/7 statements.
# Partially parsed test_thaw_tuple. Retrieved 5/6 statements.
# Partially parsed test_thaw_tuple_recursive. Retrieved 3/10 statements.
# Partially parsed test_thaw_nested_mixed. Retrieved 6/22 statements.
# Partially parsed test_thaw_strict_false_pvector. Retrieved 4/7 statements.
# Partially parsed test_thaw_strict_false_list. Retrieved 6/7 statements.
# Partially parsed test_thaw_strict_false_dict. Retrieved 5/6 statements.
# Partially parsed test_thaw_strict_false_pset. Retrieved 3/6 statements.
# Partially parsed test_thaw_strict_true_list_recursive. Retrieved 3/7 statements.
# Partially parsed test_thaw_strict_true_dict_recursive. Retrieved 4/8 statements.
# Partially parsed test_thaw_strict_false_list_no_recursion. Retrieved 3/9 statements.
# Partially parsed test_thaw_strict_false_dict_no_recursion. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = [var_0, var_4]


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
    var_2 = [var_0, var_1]
    var_3 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.thaw(var_3)
    var_5 = bool(var_4 == (1, 2, 3))
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_4]
    var_6 = 0
    var_7 = 'a'


def test_case_0():
    var_0 = 42
    var_1 = module_0.thaw(var_0)
    assert var_1 == 42
    var_2 = 'hello'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'hello'

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
    var_4 = False
    var_5 = module_0.thaw(var_3, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = False
    var_5 = module_1.thaw(var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.thaw(var_2, var_3)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = module_0.thaw(var_2, var_3)
    var_5 = bool(var_4 == (1, 2))
    assert var_5 is True

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = False
    var_5 = [var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 10/12 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 6/11 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 5/7 statements.
# Partially parsed test_freeze_strict_true_with_pmap. Retrieved 9/12 statements.
# Partially parsed test_freeze_strict_true_with_pvector. Retrieved 7/12 statements.
# Partially parsed test_freeze_mixed_container. Retrieved 18/21 statements.


import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = (var_0, var_1, var_2)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


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

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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
    var_16 = [var_9, var_10]
    var_17 = module_1.pset(var_16)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict_input. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set_input. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple_input. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 2/7 statements.
# Partially parsed test_mutant_decorator_with_positional_and_keyword_arguments. Retrieved 3/8 statements.


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
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 4/10 statements.
# Partially parsed test_freeze_strict_false_with_pvector. Retrieved 6/9 statements.
# Partially parsed test_freeze_list_with_dict. Retrieved 13/14 statements.


import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = (var_0, var_1, var_2)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_0, var_1]

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = [var_0, var_1, var_2]

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 2/8 statements.
# Partially parsed test_mutant_decorator_with_multiple_args. Retrieved 6/14 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 5/12 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 8/19 statements.
# Partially parsed test_mutant_decorator_with_frozen_inputs. Retrieved 4/9 statements.
# Partially parsed test_mutant_decorator_returns_frozen_output_for_non_container. Retrieved 1/5 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = bool(var_1 == [1])
    assert var_2 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = bool(var_2 == [1, 2])
    assert var_6 is True
    var_7 = bool(var_5 == [3, 4])
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = bool(var_2 == {'a': 1})
    assert var_5 is True

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 'original'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = {var_4, var_5}
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = bool(var_7 == {'list': ['original'], 'set': {1, 2}})
    assert var_8 is True
    var_9 = 99

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = {}
    var_1 = 'inner'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_freeze_dict_with_values. Retrieved 9/12 statements.
# Partially parsed test_freeze_list. Retrieved 6/7 statements.
# Partially parsed test_freeze_nested_list. Retrieved 10/14 statements.
# Partially parsed test_freeze_nested_tuple. Retrieved 7/9 statements.
# Partially parsed test_freeze_defaultdict. Retrieved 3/10 statements.
# Partially parsed test_freeze_pmap_strict. Retrieved 9/12 statements.
# Partially parsed test_freeze_pvector_strict. Retrieved 10/16 statements.
# Partially parsed test_freeze_nested_dict. Retrieved 9/14 statements.
# Partially parsed test_freeze_mixed_structure. Retrieved 17/22 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_1.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = (var_0, var_1, var_2)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]

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
    var_2 = [var_1]

import pyrsistent._helpers as module_1
import pyrsistent._pmap as module_0


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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = [var_7, var_1]
    var_9 = [var_3, var_4]

import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 42
    var_1 = module_0.freeze(var_0)
    var_2 = 42
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.freeze(var_0)
    var_2 = 'hello'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


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

import pyrsistent._pmap as module_1


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
    var_12 = module_0.freeze(var_11)
    var_13 = {var_3: var_4}
    var_14 = module_1.pmap(var_13)
    var_15 = [var_2, var_14]
    var_16 = [var_8]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 13/26 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/11 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_non_container_arguments. Retrieved 2/6 statements.
# Partially parsed test_mutant_decorator_with_strict_freeze. Retrieved 10/15 statements.


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
    var_12 = 'value'
    var_13 = {var_4: var_0, var_11: var_12}
    var_14 = module_0.pmap(var_13)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_1]
    var_3 = [var_1, var_0]

def test_case_0():
    var_0 = 2
    var_1 = 3


def test_case_0():
    var_0 = 'old'
    var_1 = 50
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'new'
    var_5 = 100
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_3 == var_9)
    assert var_10 is True



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 2/7 statements.
# Partially parsed test_mutant_decorator_with_mixed_arguments. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 8/22 statements.
# Partially parsed test_mutant_decorator_with_strict_false_behavior. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_handles_non_container_arguments. Retrieved 2/5 statements.


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
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    pass

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_2, var_3, var_4}
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = bool(var_7 == {'list': [1, 2, 3], 'set': {1, 2, 3}})
    assert var_8 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/9 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/13 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/21 statements.
# Failed to parse test_mutant_decorator_with_no_arguments.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 3/12 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 7/12 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_mixed_args. Retrieved 17/31 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 1/4 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 13/23 statements.
# Partially parsed test_mutant_decorator_freezes_defaultdict. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = bool(var_3 == [1, 2, 3])
    assert var_5 is True


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
    var_9 = 'new'
    var_10 = 'value'
    var_11 = {var_4: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 5
    var_14 = [var_0, var_1, var_2, var_13]
    var_15 = module_1.pset(var_14)
    var_16 = bool(var_3 == [1, 2, 3])
    assert var_16 is True
    var_17 = bool(var_5 == {'a': 1})
    assert var_17 is True
    var_18 = bool(var_6 == {1, 2, 3})
    assert var_18 is True
    var_19 = 0

def test_case_0():
    pass

def test_case_0():
    var_0 = 5

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
    var_10 = 100
    var_11 = [var_10, var_3, var_4]
    var_12 = (var_6, var_7)
    var_13 = bool(var_9 == {'list': [1, 2, 3], 'tuple': (4, 5)})
    assert var_13 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 10
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_kwargs. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 8/16 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_set. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 4/9 statements.
# Failed to parse test_mutant_decorator_with_empty_args.
# Partially parsed test_mutant_decorator_mutation_isolated. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(var_3 == [1, 2, 3])
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
    var_0 = 0
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = [var_0, var_5, var_6]
    var_8 = bool(var_7 == [0, {'x': 1, 'y': 2}, 3])
    assert var_8 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = bool(var_3 == {1, 2, 3})
    assert var_5 is True
    var_6 = 4
    var_7 = 1
    var_8 = 2
    var_9 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = bool(var_3 == (1, 2, 3))
    assert var_4 is True
    var_5 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = bool(var_1 == [1])
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_mutant_decorator_does_not_freeze_return_value_when_it_is_already_frozen. Retrieved 8/16 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_1, var_4, var_5]
    var_7 = module_1.pset(var_6)



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_mutant_with_positional_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_with_keyword_arguments. Retrieved 7/12 statements.
# Partially parsed test_mutant_with_mixed_arguments. Retrieved 14/23 statements.
# Partially parsed test_mutant_returns_frozen_result. Retrieved 5/12 statements.
# Partially parsed test_mutant_with_nested_mutables_in_args. Retrieved 12/21 statements.
# Failed to parse test_mutant_preserves_function_metadata.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_2 == {'a': 1})
    assert var_7 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 5
    var_6 = {var_5}
    var_7 = 99
    var_8 = [var_0, var_7]
    var_9 = 'new'
    var_10 = {var_2: var_3, var_9: var_7}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_5, var_7}
    var_13 = module_1.pset(var_12)
    var_14 = bool(var_1 == [1])
    assert var_14 is True
    var_15 = bool(var_4 == {'x': 10})
    assert var_15 is True
    var_16 = bool(var_6 == {5})
    assert var_16 is True

def test_case_0():
    var_0 = 'inner'
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
    var_5 = {var_4}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'changed'
    var_8 = [var_7]
    var_9 = 100
    var_10 = {var_4, var_9}
    var_11 = module_0.pset(var_10)
    var_12 = bool(var_6 == {'list': ['original'], 'set': {1}})
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_mutant_decorator_freezes_args_and_return. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_kwargs. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_freezes_nested_structures. Retrieved 11/27 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_preserves_non_container_return. Retrieved 3/8 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 6/11 statements.
# Partially parsed test_mutant_decorator_freezes_defaultdict. Retrieved 2/9 statements.
# Partially parsed test_mutant_decorator_with_multiple_args. Retrieved 7/14 statements.


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
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = (var_5, var_8)
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = bool(var_10 == {'list': [1, 2], 'tuple': (3, [4, 5])})
    assert var_11 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = bool(var_3 == {1, 2, 3})
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 'hello'
    var_2 = None

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    assert var_6 == 5
    var_7 = bool(var_2 == [1, 2])
    assert var_7 is True
    var_8 = bool(var_5 == [3, 4])
    assert var_8 is True



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_mutant_decorator_freezes_arguments_and_return. Retrieved 12/29 statements.


import pyrsistent._helpers as module_1


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
    var_13 = 'new_value'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'new_key'
    var_17 = {var_14: var_0, var_15: var_1, var_16: var_13}
    var_18 = module_0.m(**var_17)
    var_19 = {}
    var_20 = module_1.freeze(var_19)
    var_21 = [var_20]
    var_22 = [var_0, var_1, var_6]
    var_23 = 999
    var_24 = [var_0, var_1, var_6, var_23]
    var_25 = set()
    var_26 = module_1.freeze(var_25)
    var_27 = [var_26]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_mutant_decorator_does_not_mutate_inputs. Retrieved 8/21 statements.



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



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/20 statements.
# Partially parsed test_mutant_decorator_with_no_mutation. Retrieved 1/4 statements.
# Partially parsed test_mutant_decorator_with_strict_false_implicitly. Retrieved 2/10 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/11 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/10 statements.
# Partially parsed test_mutant_decorator_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 6/12 statements.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 9/23 statements.
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



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mutant_decorator_freezes_inputs_and_output. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_dict. Retrieved 5/12 statements.
# Partially parsed test_mutant_decorator_with_set. Retrieved 4/11 statements.
# Partially parsed test_mutant_decorator_with_keyword_arguments. Retrieved 4/10 statements.
# Failed to parse test_mutant_decorator_preserves_function_metadata.
# Partially parsed test_mutant_decorator_with_nested_structures. Retrieved 5/15 statements.
# Failed to parse test_mutant_decorator_with_empty_arguments.
# Partially parsed test_mutant_decorator_freezes_arguments_recursively. Retrieved 7/13 statements.


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
    var_3 = 'key_'
    var_4 = bool(var_2 == {'x': 10})
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = bool(var_4 == {'a': [1, 2]})
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(var_6 == [[1, 2], {'a': 3}])
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_freeze_defaultdict_with_strict_true. Retrieved 9/13 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = [var_5, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.pmap(var_7)



