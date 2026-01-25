####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_var_positional.
# Failed to parse test_get_arity_with_var_keyword.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_mixed_parameters.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_one_positional_parameter.
# Failed to parse test_get_arity_with_multiple_positional_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameter.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_varkwargs.
# Failed to parse test_get_arity_with_positional_and_keyword_only.
# Failed to parse test_get_arity_with_default_parameter.
# Failed to parse test_get_arity_with_all_default_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameter.
# Failed to parse test_get_arity_with_positional_only_parameter.
# Failed to parse test_get_arity_with_mixed_parameter_types.




# Parsed testcases at query #3
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 6)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #4
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(2, 7)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #5
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [('a', 1), ('b', 2)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 5), (1, 15), (2, 25)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [(0, 99)])
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'c'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/53 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #8
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 6)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #9
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [('a', 1), ('b', 2)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [(0, 99)])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'ab'
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [(0, 'a'), (1, 'b')])
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_one_positional_parameter.
# Failed to parse test_get_arity_with_multiple_positional_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameter.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_varkwargs.
# Failed to parse test_get_arity_with_positional_and_keyword_only.
# Failed to parse test_get_arity_with_default_parameter.
# Failed to parse test_get_arity_with_all_default_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameter.
# Failed to parse test_get_arity_with_positional_only_parameter.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'c'



# Parsed testcases at query #12
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]
    var_9 = list(var_5)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_6, var_8, var_10]
    var_12 = list(var_4)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = []
    var_3 = list(var_1)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_6, var_8, var_10]
    var_12 = list(var_4)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = 0
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = 1
    var_6 = 'b'
    var_7 = (var_5, var_6)
    var_8 = 2
    var_9 = 'c'
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = list(var_1)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = lambda k, v: v % var_4 == var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_1, var_4)
    var_11 = [var_10]
    var_12 = bool(var_9 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 4/12 statements.
# Partially parsed test_update_structure_discard_nested. Retrieved 5/14 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_non_existing_key. Retrieved 3/9 statements.
# Partially parsed test_update_structure_set_new_leaf. Retrieved 4/10 statements.
# Partially parsed test_update_structure_set_nested_new. Retrieved 3/12 statements.
# Partially parsed test_update_structure_update_existing_leaf. Retrieved 7/11 statements.
# Partially parsed test_update_structure_update_nested_existing. Retrieved 7/16 statements.
# Partially parsed test_update_structure_with_empty_pmap_leaf. Retrieved 2/11 statements.
# Partially parsed test_update_structure_no_change. Retrieved 7/11 statements.
# Partially parsed test_update_structure_discard_reverse_order. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_callable_command. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'x'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 10
    var_2 = lambda x: var_1
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 5
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = lambda x: x + var_0
    var_5 = []
    var_6 = 2

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = [var_3]
    var_5 = lambda x: x
    var_6 = []

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = 2
    var_8 = (var_7, var_2)
    var_9 = [var_4, var_6, var_8]
    var_10 = []

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = '5'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_value_error. Retrieved 1/5 statements.
# Partially parsed test_predicate_with_three_args_raises_value_error. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key_missing. Retrieved 6/8 statements.
# Partially parsed test_get_keys_and_values_with_sequence_and_non_callable_key_out_of_range. Retrieved 6/8 statements.
# Partially parsed test_get_keys_and_values_with_object_and_non_callable_key. Retrieved 6/9 statements.
# Partially parsed test_get_keys_and_values_with_object_and_non_callable_key_missing. Retrieved 2/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = sorted(var_9)
    var_14 = sorted(var_12)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = sorted(var_8)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 2
    var_6 = lambda k: k == var_4 or k == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = (var_4, var_0)
    var_9 = (var_5, var_2)
    var_10 = [var_8, var_9]
    var_11 = sorted(var_7)
    var_12 = sorted(var_10)
    var_13 = bool(var_11 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 'b'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 5

def test_case_0():
    var_0 = 42
    var_1 = 'x'
    var_2 = 'x'
    var_3 = 42
    var_4 = (var_2, var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 'y'
    var_1 = 'y'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 2/48 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/10 statements.
# Partially parsed test__do_to_path_with_missing_key_and_discard. Retrieved 7/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = lambda x: sum(x)
    var_6 = module_0._do_to_path(var_3, var_4, var_5)
    assert var_6 == 6

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0._do_to_path(var_3, var_4, var_8)
    var_10 = bool(var_9 == [4, 5, 6])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x * var_3
    var_7 = module_0._do_to_path(var_4, var_5, var_6)
    var_8 = bool(var_7 == {'a': 2, 'b': 2})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k % var_2 == var_0
    var_8 = [var_7]
    var_9 = lambda x: x.upper()
    var_10 = module_0._do_to_path(var_6, var_8, var_9)
    var_11 = bool(var_10 == {0: 'A', 1: 'b', 2: 'C'})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 15
    var_8 = lambda k, v: v > var_7
    var_9 = [var_8]
    var_10 = 2
    var_11 = lambda x: x * var_10
    var_12 = module_0._do_to_path(var_6, var_9, var_11)
    var_13 = bool(var_12 == {'x': 10, 'y': 40, 'z': 60})
    assert var_13 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_1]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = [var_0, var_3]
    var_12 = 10
    var_13 = lambda x: x * var_12
    var_14 = module_0._do_to_path(var_10, var_11, var_13)
    var_15 = bool(var_14 == {'a': {'x': 1, 'y': 20}, 'b': {'x': 3, 'y': 4}})
    assert var_15 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = [var_5]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = 99
    var_8 = lambda x: var_7
    var_9 = module_0._do_to_path(var_4, var_6, var_8)
    var_10 = bool(var_9 == {'a': 1, 'b': 2, 'c': 99})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_4]
    var_6 = 5
    var_7 = lambda x: x + var_6
    var_8 = module_0._do_to_path(var_3, var_5, var_7)
    var_9 = bool(var_8 == [10, 25, 30])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 1
    var_6 = lambda k: k > var_5
    var_7 = [var_6]
    var_8 = 2
    var_9 = lambda x: x * var_8
    var_10 = module_0._do_to_path(var_4, var_7, var_9)
    var_11 = bool(var_10 == [5, 10, 30, 40])
    assert var_11 is True



# Parsed testcases at query #20
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = var_1(var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'test_'
    var_6 = var_1(var_5)
    assert var_6 is None
    var_7 = '123'
    var_8 = var_1(var_7)
    assert var_8 is None
    var_9 = 'test_abc'
    var_10 = var_1(var_9)
    assert var_10 is None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test$'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False
    var_4 = None
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = []
    var_7 = var_1(var_6)
    assert var_7 is False
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = var_1(var_10)
    assert var_11 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123_extra'
    var_3 = var_1(var_2)
    assert var_3 is None
    var_4 = 'prefix_test_123'
    var_5 = var_1(var_4)
    assert var_5 is None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = 'any_string'
    var_5 = var_1(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\.b$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a.b'
    var_3 = var_1(var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'aXb'
    var_6 = var_1(var_5)
    assert var_6 is None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^Test$'
    var_1 = module_0.rex(var_0)
    var_2 = 'Test'
    var_3 = var_1(var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'test'
    var_6 = var_1(var_5)
    assert var_6 is None



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [('a', 1), ('b', 2)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 5), (1, 15), (2, 25)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [(0, 99)])
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 3/40 statements.


def test_case_0():
    var_0 = 'param2'
    var_1 = 'param1'
    var_2 = 'args'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_varkwargs.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_mixed_parameters.
# Failed to parse test_get_arity_with_builtin_function.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_with_arity_other_than_1_or_2. Retrieved 1/5 statements.
# Partially parsed test_predicate_with_arity_three. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #25
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda k: k % var_0 == var_1
    var_3 = 1
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_1: var_4, var_3: var_5, var_0: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_1, var_4)
    var_10 = (var_0, var_6)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = lambda k, v: v == var_0
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 'c'
    var_7 = {var_2: var_5, var_3: var_0, var_4: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = (var_3, var_0)
    var_10 = [var_9]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_0)
    var_9 = (var_2, var_5)
    var_10 = [var_9]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_2: var_3, var_0: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 0
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_2: var_3, var_0: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda i: i > var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = 1
    var_8 = (var_7, var_3)
    var_9 = 2
    var_10 = (var_9, var_4)
    var_11 = [var_8, var_10]
    var_12 = bool(var_6 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'c'
    var_1 = lambda i, v: v == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3, var_0]
    var_5 = module_0._get_keys_and_values(var_4, var_1)
    var_6 = 2
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_error. Retrieved 3/7 statements.
# Partially parsed test_predicate_with_three_args_raises_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_and_attribute_key. Retrieved 2/5 statements.
# Partially parsed test__get_keys_and_values_with_object_and_non_existent_attribute_key. Retrieved 2/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)
    var_6 = bool(var_5 == [('a', 1)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0._get_keys_and_values(var_3, var_4)

def test_case_0():
    var_0 = 100
    var_1 = 'x'

def test_case_0():
    var_0 = 100
    var_1 = 'y'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = sorted(var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__get_keys_and_values_with_missing_non_callable_key_in_dict. Retrieved 6/8 statements.
# Partially parsed test__get_keys_and_values_with_missing_non_callable_key_in_list. Retrieved 6/8 statements.
# Partially parsed test__get_keys_and_values_with_object_and_non_callable_key. Retrieved 6/9 statements.
# Partially parsed test__get_keys_and_values_with_object_and_missing_non_callable_key. Retrieved 2/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = sorted(var_9)
    var_14 = sorted(var_12)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = sorted(var_8)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = (var_5, var_0)
    var_9 = (var_4, var_2)
    var_10 = [var_8, var_9]
    var_11 = sorted(var_7)
    var_12 = sorted(var_10)
    var_13 = bool(var_11 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_8, var_10]
    var_12 = sorted(var_6)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 'b'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 5

def test_case_0():
    var_0 = 42
    var_1 = 'attr'
    var_2 = 'attr'
    var_3 = 42
    var_4 = (var_2, var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 'missing'
    var_1 = 'missing'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_and_callable_unary. Retrieved 4/7 statements.
# Partially parsed test__get_keys_and_values_with_object_and_callable_binary. Retrieved 4/7 statements.
# Partially parsed test__get_keys_and_values_with_object_and_non_callable_key. Retrieved 2/5 statements.
# Partially parsed test__get_keys_and_values_with_object_missing_attribute_returns_sentinel. Retrieved 2/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)
    var_6 = bool(var_5 == [('a', 1)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda idx: idx % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda idx, val: val > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'x'
    var_3 = lambda attr: attr == var_2

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 10
    var_3 = lambda attr, val: val == var_2

def test_case_0():
    var_0 = 5
    var_1 = 'x'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_0():
    var_0 = 5
    var_1 = 'y'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_and_callable_unary. Retrieved 2/8 statements.
# Partially parsed test__get_keys_and_values_with_object_and_callable_binary. Retrieved 2/8 statements.
# Partially parsed test__get_keys_and_values_with_object_and_non_callable_key. Retrieved 1/6 statements.
# Partially parsed test__get_keys_and_values_with_object_missing_non_callable_key_returns_sentinel. Retrieved 1/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = sorted(var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda i: i % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_0():
    var_0 = 'x'
    var_1 = lambda k: k == var_0

def test_case_0():
    var_0 = 10
    var_1 = lambda k, v: v == var_0

def test_case_0():
    var_0 = 'x'

def test_case_0():
    var_0 = 'z'



# Parsed testcases at query #32
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = [var_9]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_structure_discard_with_empty_path. Retrieved 6/10 statements.
# Partially parsed test_update_structure_discard_with_nested_path. Retrieved 5/14 statements.
# Partially parsed test_update_structure_discard_multiple_keys_reversed. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_key_not_present. Retrieved 3/9 statements.
# Partially parsed test_update_structure_update_with_new_value. Retrieved 7/11 statements.
# Partially parsed test_update_structure_update_with_nested_path. Retrieved 6/15 statements.
# Partially parsed test_update_structure_expand_with_empty_sentinel. Retrieved 4/10 statements.
# Partially parsed test_update_structure_expand_nested_with_empty_sentinel. Retrieved 5/12 statements.
# Partially parsed test_update_structure_no_change_when_result_equals_value. Retrieved 6/9 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = [var_3]
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_3]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 2
    var_6 = (var_5, var_2)
    var_7 = [var_4, var_6]
    var_8 = []

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = 100
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 99
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'new'
    var_1 = []
    var_2 = 42
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = 5
    var_4 = lambda x: var_3

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = lambda x: x

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x * var_8
    var_10 = 20



# Parsed testcases at query #34
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v == var_4
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda i: i == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda i, v: v == var_2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 5
    var_4 = 10
    var_5 = {var_0: var_3, var_1: var_4, var_2: var_3}
    var_6 = lambda k, v: v == var_3
    var_7 = module_0._get_keys_and_values(var_5, var_6)
    var_8 = bool(var_7 == [('x', 5), ('z', 5)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 4/12 statements.
# Partially parsed test_update_structure_discard_nested. Retrieved 4/15 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_non_existent_key. Retrieved 3/9 statements.
# Partially parsed test_update_structure_update_leaf. Retrieved 7/11 statements.
# Partially parsed test_update_structure_update_nested. Retrieved 6/15 statements.
# Partially parsed test_update_structure_insert_new_empty_pmap. Retrieved 3/12 statements.
# Partially parsed test_update_structure_no_change. Retrieved 6/9 statements.
# Partially parsed test_update_structure_with_callable_command. Retrieved 11/20 statements.
# Partially parsed test_update_structure_discard_reverse_order. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = (var_3, var_0)
    var_5 = 'b'
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = []

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = lambda x: x + var_0
    var_5 = []
    var_6 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = lambda x: x + var_0
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = lambda x: x
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 10
    var_5 = 'c'
    var_6 = 20
    var_7 = lambda x: m(b=x[var_3] + var_4, c=x[var_5] + var_6)
    var_8 = []
    var_9 = 11
    var_10 = 22

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 2
    var_6 = (var_5, var_2)
    var_7 = [var_4, var_6]
    var_8 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_discard_command_and_nested_path. Retrieved 9/21 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_discard_command. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_non_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_multiple_kvs_and_discard_command_reversed. Retrieved 13/18 statements.
# Partially parsed test_update_structure_with_nested_structure_and_path. Retrieved 10/27 statements.
# Partially parsed test_update_structure_with_result_equal_to_original_value. Retrieved 7/10 statements.
# Partially parsed test_update_structure_with_empty_kvs_list. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = {var_2: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x + var_1
    var_7 = 2
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 100
    var_7 = 100
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 5
    var_6 = lambda x: var_5
    var_7 = {var_0: var_1, var_3: var_5}

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = (var_2, var_5)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = [var_1, var_2]
    var_7 = lambda x: x + var_3
    var_8 = 2
    var_9 = {var_2: var_8}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = []



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__do_to_path_with_discard_command_on_missing_key. Retrieved 7/8 statements.
# Partially parsed test__do_to_path_with_discard_command_on_existing_key. Retrieved 6/7 statements.
# Partially parsed test__do_to_path_with_discard_command_on_list_structure. Retrieved 6/7 statements.
# Partially parsed test__do_to_path_with_callable_key_spec_and_discard. Retrieved 9/10 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = lambda x: sum(x)
    var_6 = module_0._do_to_path(var_3, var_4, var_5)
    assert var_6 == 6

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'new_value'
    var_5 = module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 'new_value'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x * var_3
    var_7 = module_0._do_to_path(var_4, var_5, var_6)
    var_8 = bool(var_7 == {'a': 2, 'b': 2})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 100
    var_7 = module_0._do_to_path(var_4, var_5, var_6)
    var_8 = bool(var_7 == {'a': 100, 'b': 2})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k % var_2 == var_0
    var_8 = [var_7]
    var_9 = lambda x: x.upper()
    var_10 = module_0._do_to_path(var_6, var_8, var_9)
    var_11 = bool(var_10 == {0: 'A', 1: 'b', 2: 'C'})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = [var_7]
    var_9 = 10
    var_10 = lambda x: x * var_9
    var_11 = module_0._do_to_path(var_6, var_8, var_10)
    var_12 = bool(var_11 == {'a': 1, 'b': 20, 'c': 30})
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = [var_0, var_2]
    var_12 = 100
    var_13 = lambda v: v + var_12
    var_14 = module_0._do_to_path(var_10, var_11, var_13)
    var_15 = bool(var_14 == {'a': {'x': 101, 'y': 2}, 'b': {'x': 3, 'y': 4}})
    assert var_15 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_4]

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'new_key'
    var_2 = 'nested_key'
    var_3 = [var_1, var_2]
    var_4 = 'value'
    var_5 = module_0._do_to_path(var_0, var_3, var_4)
    var_6 = {var_2: var_4}
    var_7 = {var_1: var_6}
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = [var_4]
    var_6 = 10
    var_7 = lambda x: x / var_6
    var_8 = module_0._do_to_path(var_3, var_5, var_7)
    var_9 = bool(var_8 == [10, 200, 300])
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_keys_and_values_with_missing_key_in_mapping. Retrieved 6/8 statements.
# Partially parsed test_get_keys_and_values_with_missing_index_in_sequence. Retrieved 5/7 statements.
# Partially parsed test_get_keys_and_values_with_object_and_attribute. Retrieved 6/9 statements.
# Partially parsed test_get_keys_and_values_with_object_and_missing_attribute. Retrieved 2/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = sorted(var_9)
    var_14 = sorted(var_12)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = sorted(var_8)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = lambda k: k in var_6
    var_8 = module_0._get_keys_and_values(var_3, var_7)
    var_9 = (var_4, var_0)
    var_10 = (var_5, var_2)
    var_11 = [var_9, var_10]
    var_12 = sorted(var_8)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_8, var_10]
    var_12 = sorted(var_6)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 'b'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = 5

def test_case_0():
    var_0 = 42
    var_1 = 'attr'
    var_2 = 'attr'
    var_3 = 42
    var_4 = (var_2, var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 'missing'
    var_1 = 'missing'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda a, b, c: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_one_positional_parameter.
# Failed to parse test_get_arity_with_multiple_positional_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameter.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_varkwargs.
# Failed to parse test_get_arity_with_mixed_parameters_and_defaults.
# Failed to parse test_get_arity_with_positional_only_parameters.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_error. Retrieved 3/7 statements.
# Partially parsed test_predicate_with_three_args_raises_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_one_positional_parameter.
# Failed to parse test_get_arity_with_multiple_positional_parameters.
# Failed to parse test_get_arity_with_positional_and_keyword_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_varkwargs.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_mixed_parameter_kinds.




# Parsed testcases at query #6
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [('a', 1), ('b', 2)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [(0, 99)])
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_having_items. Retrieved 2/7 statements.
# Partially parsed test__get_keys_and_values_with_object_having_getitem. Retrieved 5/14 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(2, 30)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)
    var_6 = bool(var_5 == [('x', 100)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(2, 7)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_0():
    var_0 = '2'
    var_1 = lambda k, v: var_0 in k

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = [var_0, var_1, var_2]
    var_4 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_callable_with_arity_0. Retrieved 1/5 statements.
# Partially parsed test_callable_with_arity_3. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = sorted(var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = 'third'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'second')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_callable_with_arity_0_raises_value_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = lambda k: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 2/88 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #13
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = bool(var_5 == [('a', 1), ('b', 2)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 10), (1, 20), (2, 30)])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 100), (1, 200), (2, 300)])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = [var_9]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda i: i % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = (var_5, var_0)
    var_9 = (var_4, var_2)
    var_10 = [var_8, var_9]
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_8, var_10]
    var_12 = bool(var_6 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 6
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 6
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(2, 7)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_arity_other_than_1_or_2_raises_value_error. Retrieved 1/6 statements.
# Partially parsed test_predicate_with_three_args_raises_value_error. Retrieved 1/6 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 2/88 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__get_keys_and_values_with_non_existent_key_returns_empty_sentinel. Retrieved 6/8 statements.
# Partially parsed test__get_keys_and_values_with_object_and_attribute. Retrieved 6/9 statements.
# Partially parsed test__get_keys_and_values_with_object_and_missing_attribute_returns_empty_sentinel. Retrieved 2/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = sorted(var_9)
    var_14 = sorted(var_12)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = sorted(var_8)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = (var_5, var_0)
    var_9 = (var_4, var_2)
    var_10 = [var_8, var_9]
    var_11 = sorted(var_7)
    var_12 = sorted(var_10)
    var_13 = bool(var_11 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_8, var_10]
    var_12 = sorted(var_6)
    var_13 = sorted(var_11)
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda a, b, c: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 'b'

def test_case_0():
    var_0 = 42
    var_1 = 'attr'
    var_2 = 'attr'
    var_3 = 42
    var_4 = (var_2, var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 'missing_attr'
    var_1 = 'missing_attr'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/53 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda k: k > var_0
    var_2 = 1
    var_3 = -2
    var_4 = 3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0._get_keys_and_values(var_8, var_1)
    var_10 = (var_2, var_5)
    var_11 = (var_4, var_7)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = lambda k, v: k > var_0 and v.startswith(var_1)
    var_3 = 1
    var_4 = -2
    var_5 = 3
    var_6 = 'apple'
    var_7 = 'banana'
    var_8 = 'apricot'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0._get_keys_and_values(var_9, var_2)
    var_11 = (var_3, var_6)
    var_12 = (var_5, var_8)
    var_13 = [var_11, var_12]
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda i: i % var_0 == var_1
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_1, var_3)
    var_10 = (var_0, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda i, v: i % var_0 == var_1 and v.isupper()
    var_3 = 'A'
    var_4 = 'b'
    var_5 = 'C'
    var_6 = 'd'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_1, var_3)
    var_10 = (var_0, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = 2
    var_8 = (var_7, var_3)
    var_9 = [var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/53 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_value_error. Retrieved 1/5 statements.
# Partially parsed test_predicate_with_three_arguments_raises_value_error. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_arity_other_than_1_or_2. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #26
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 6)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda k: k > var_0
    var_2 = 1
    var_3 = -2
    var_4 = 3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0._get_keys_and_values(var_8, var_1)
    var_10 = (var_2, var_5)
    var_11 = (var_4, var_7)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = lambda k, v: k > var_0 and v.startswith(var_1)
    var_3 = 1
    var_4 = -2
    var_5 = 3
    var_6 = 'apple'
    var_7 = 'banana'
    var_8 = 'apricot'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0._get_keys_and_values(var_9, var_2)
    var_11 = (var_3, var_6)
    var_12 = (var_5, var_8)
    var_13 = [var_11, var_12]
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = module_0._get_keys_and_values(var_3, var_1)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = module_0._get_keys_and_values(var_3, var_1)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = enumerate(var_5)
    var_7 = list(var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_arity_0. Retrieved 3/7 statements.
# Partially parsed test_get_keys_and_values_with_callable_arity_3. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('x', 100)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 6)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #31
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*z$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcz'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*z$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcy'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False
    var_4 = None
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = []
    var_7 = var_1(var_6)
    assert var_7 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a.*z'
    var_1 = module_0.rex(var_0)
    var_2 = 'a test z'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'start a test z end'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_0)
    assert var_2 is True
    var_3 = 'any'
    var_4 = var_1(var_3)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+\\.\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123.456'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = '123'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = 'abc.def'
    var_7 = var_1(var_6)
    assert var_7 is False



# Parsed testcases at query #32
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = (var_2, var_5)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda k, v: v % var_1 == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = [var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]
    var_9 = sorted(var_5)
    var_10 = sorted(var_8)
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_6, var_8, var_10]
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = 1
    var_8 = (var_7, var_1)
    var_9 = 2
    var_10 = (var_9, var_2)
    var_11 = [var_6, var_8, var_10]
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = []
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = []
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'ab'
    var_1 = module_0._items(var_0)
    var_2 = 0
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = 1
    var_6 = 'b'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = bool(var_1 == var_8)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda k: k % var_0 == var_1
    var_3 = 1
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_1: var_4, var_3: var_5, var_0: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_1, var_4)
    var_10 = (var_0, var_6)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'b'
    var_2 = lambda k, v: k > var_0 and v.startswith(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 'bc'
    var_7 = {var_0: var_5, var_3: var_1, var_4: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_3, var_1)
    var_10 = (var_4, var_6)
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 42
    var_4 = 0
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = {}
    var_3 = module_0._get_keys_and_values(var_2, var_1)
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda i: i == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = (var_0, var_3)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 10/15 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_discard_command_and_non_empty_path. Retrieved 13/29 statements.
# Partially parsed test_update_structure_with_callable_command_and_non_empty_path. Retrieved 16/31 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_discard_command. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_non_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_nested_path. Retrieved 11/22 statements.
# Partially parsed test_update_structure_with_reversed_discard_for_sequence. Retrieved 13/18 statements.
# Partially parsed test_update_structure_with_no_change_when_result_equals_original. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = lambda x: x * var_3
    var_10 = 4
    var_11 = {var_0: var_3, var_1: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = 100
    var_10 = 100
    var_11 = {var_0: var_10, var_1: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 20
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_3}
    var_9 = {var_5: var_6}
    var_10 = [var_2]
    var_11 = {}
    var_12 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 20
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_3}
    var_9 = {var_5: var_6}
    var_10 = [var_2]
    var_11 = 5
    var_12 = lambda x: x + var_11
    var_13 = 15
    var_14 = {var_2: var_13}
    var_15 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 99
    var_6 = 99
    var_7 = {var_0: var_1, var_3: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = 'y'
    var_6 = [var_5]
    var_7 = 50
    var_8 = {var_1: var_2}
    var_9 = 50
    var_10 = {var_5: var_9}

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = (var_4, var_0)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = 2
    var_9 = (var_8, var_2)
    var_10 = [var_5, var_7, var_9]
    var_11 = []
    var_12 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = lambda x: x



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = []



# Parsed testcases at query #37
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = lambda k: k == var_0
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = (var_0, var_3)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda k, v: v > var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = (var_3, var_4)
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda i: i % var_0 == var_1
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0._get_keys_and_values(var_6, var_2)
    var_8 = (var_1, var_3)
    var_9 = (var_0, var_5)
    var_10 = [var_8, var_9]
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 15
    var_1 = lambda i, v: v > var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = 1
    var_8 = (var_7, var_3)
    var_9 = 2
    var_10 = (var_9, var_4)
    var_11 = [var_8, var_10]
    var_12 = bool(var_6 == var_11)
    assert var_12 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 7/18 statements.
# Partially parsed test_update_structure_discard_nonexistent. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 9/18 statements.
# Partially parsed test_update_structure_discard_reverse_order. Retrieved 11/23 statements.
# Partially parsed test_update_structure_update_leaf. Retrieved 9/20 statements.
# Partially parsed test_update_structure_update_with_empty_sentinel. Retrieved 8/19 statements.
# Partially parsed test_update_structure_update_multiple_keys. Retrieved 15/30 statements.
# Partially parsed test_update_structure_no_change. Retrieved 8/19 statements.
# Partially parsed test_update_structure_empty_path_update. Retrieved 9/13 statements.
# Partially parsed test_update_structure_empty_path_discard. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = {}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = []
    var_8 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 0
    var_7 = [var_0, var_1]
    var_8 = [var_3, var_4]
    var_9 = []
    var_10 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = {}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 1
    var_6 = lambda x: var_5
    var_7 = {var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'd'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_3}
    var_9 = {var_5: var_6}
    var_10 = [var_2]
    var_11 = 3
    var_12 = lambda x: var_11
    var_13 = {var_2: var_11}
    var_14 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = lambda x: x
    var_7 = {var_1: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = {var_0: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = {}



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_discard_with_empty_path. Retrieved 8/12 statements.
# Partially parsed test_update_structure_discard_with_nested_path. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_reverse_order_for_vectors. Retrieved 11/15 statements.
# Partially parsed test_update_structure_discard_skip_empty_sentinel. Retrieved 3/9 statements.
# Partially parsed test_update_structure_update_with_empty_sentinel_and_pmap. Retrieved 4/10 statements.
# Partially parsed test_update_structure_update_with_nested_path_and_modification. Retrieved 6/15 statements.
# Partially parsed test_update_structure_no_change_when_result_equals_original. Retrieved 6/9 statements.
# Partially parsed test_update_structure_update_multiple_keys. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'x'
    var_5 = [var_4]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = 2
    var_8 = (var_7, var_2)
    var_9 = [var_4, var_6, var_8]
    var_10 = []

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []

def test_case_0():
    var_0 = 'new_key'
    var_1 = 42
    var_2 = lambda x: var_1
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 'b'
    var_5 = [var_4]

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = lambda x: x
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = 10
    var_8 = lambda x: x + var_7
    var_9 = []
    var_10 = 11
    var_11 = 12



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = lambda x: x
    var_9 = {var_6: var_3}



