####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_single_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




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
    var_7 = lambda k: k.startswith(var_0)
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
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
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
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



# Parsed testcases at query #4
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #5
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 6/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = lambda x: x.clear() or x
    var_7 = module_0._do_to_path(var_4, var_5, var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = module_0._do_to_path(var_4, var_5, var_8)
    var_10 = bool(var_9 == {'c': 3})
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = lambda x: x * var_3
    var_8 = module_0._do_to_path(var_4, var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = [var_0, var_2]
    var_8 = lambda x: x * var_5
    var_9 = module_0._do_to_path(var_6, var_7, var_8)
    var_10 = bool(var_9 == {'a': {'b': 2}, 'c': 2})
    assert var_10 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k != var_1
    var_8 = [var_7]
    var_9 = lambda x: x * var_4
    var_10 = module_0._do_to_path(var_6, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'c': 6, 'b': 2})
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
    var_9 = lambda x: x * var_4
    var_10 = module_0._do_to_path(var_6, var_8, var_9)
    var_11 = bool(var_10 == {'a': 1, 'b': 4, 'c': 6})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k, v, w: var_5
    var_7 = [var_6]
    var_8 = 2
    var_9 = lambda x: x * var_8
    var_10 = module_0._do_to_path(var_4, var_7, var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #7
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_7 = 'd'
    var_8 = module_0._get_keys_and_values(var_6, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True
    var_8 = lambda k, v, x: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
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
    var_5 = True
    var_6 = lambda k, v, x: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)



# Parsed testcases at query #10
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
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
    var_7 = True
    var_8 = lambda x, y, z: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'd'
    var_8 = module_0._get_keys_and_values(var_6, var_7)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_get_arity_with_no_required_params.




# Parsed testcases at query #12
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k: k == var_0
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True



# Parsed testcases at query #15
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'p'), (1, 'q'), (2, 'r')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_items_without_items_method. Retrieved 5/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 1), (1, 2), (2, 3)])
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #18
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True
    var_8 = lambda k, v, x: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_7 = 'd'
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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



# Parsed testcases at query #19
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k: k == var_0
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Failed to parse test__get_arity_with_default_parameters.




# Parsed testcases at query #22
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test_\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'test_abc'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = 123
    var_7 = var_1(var_6)
    assert var_7 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_non_discard_command. Retrieved 8/13 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_discard_command. Retrieved 7/11 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]

import pyrsistent._transformations as module_0

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
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'b': 4})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]
    var_15 = lambda x: x * var_6
    var_16 = module_0._update_structure(var_8, var_13, var_14, var_15)
    var_17 = bool(var_16 == {'a': {'x': 2}, 'b': {'y': 2}})
    assert var_17 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x * var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = 1
    var_3 = [var_2]
    var_4 = 'some_command'



# Parsed testcases at query #26
#--------------------------

# Failed to parse test__get_arity_with_default_parameters.




# Parsed testcases at query #27
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #28
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 'foo'), (1, 'bar')])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #30
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.
# Failed to parse test_get_arity_with_mixed_args.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = []
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'some_command'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #33
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
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
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
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



# Parsed testcases at query #34
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/5 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 13/15 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/12 statements.
# Partially parsed test_update_structure_with_pmap_leaf_node. Retrieved 8/13 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]

import pyrsistent._transformations as module_0

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
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'b': 4})
    assert var_11 is True

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
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]
    var_13 = lambda x: x * var_5
    var_14 = module_0._update_structure(var_8, var_11, var_12, var_13)
    var_15 = bool(var_14 == {'a': {'x': 2, 'y': 2}, 'b': 3})
    assert var_15 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x

import pyrsistent._transformations as module_0

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
    var_9 = 0
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 0, 'b': 0})
    assert var_11 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/10 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 11/12 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_non_discard_command. Retrieved 6/9 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_discard_command. Retrieved 5/8 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_2: var_3}
    var_8 = (var_0, var_7)
    var_9 = [var_8]
    var_10 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = lambda x: x
    var_6 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1: var_2}
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = lambda x: x + var_2
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': {'b': 2}})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x
    var_7 = module_0._update_structure(var_2, var_4, var_5, var_6)
    var_8 = bool(var_7 == {'a': 1})
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_non_discard_command. Retrieved 8/14 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]

import pyrsistent._transformations as module_0

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
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'b': 4})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]
    var_15 = lambda x: x * var_6
    var_16 = module_0._update_structure(var_8, var_13, var_14, var_15)
    var_17 = bool(var_16 == {'a': {'x': 2}, 'b': {'y': 2}})
    assert var_17 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/10 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = lambda x: x.update(var_7) or x
    var_9 = []
    var_10 = module_0._do_to_path(var_4, var_9, var_8)
    var_11 = bool(var_10 == {'a': 1, 'b': 2, 'c': 3})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = []
    var_9 = module_0._do_to_path(var_4, var_8, var_7)
    var_10 = bool(var_9 == {'c': 3})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = lambda x: x * var_3
    var_8 = [var_0, var_2]
    var_9 = module_0._do_to_path(var_6, var_8, var_7)
    var_10 = bool(var_9 == {'a': {'b': 4}, 'c': 3})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = [var_0, var_2]
    var_9 = module_0._do_to_path(var_6, var_8, var_7)
    var_10 = bool(var_9 == {'a': {'b': 4}, 'c': 3})
    assert var_10 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = [var_0, var_2]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda x: x * var_4
    var_8 = [var_0, var_1]
    var_9 = lambda k: k in var_8
    var_10 = [var_9]
    var_11 = module_0._do_to_path(var_6, var_10, var_7)
    var_12 = bool(var_11 == {'a': 2, 'b': 4, 'c': 3})
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
    var_7 = lambda x: x * var_4
    var_8 = lambda k, v: v > var_3
    var_9 = [var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)
    var_11 = bool(var_10 == {'a': 1, 'b': 4, 'c': 6})
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------




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
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k.startswith(var_0)
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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



# Parsed testcases at query #3
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



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
    var_7 = lambda k: k.startswith(var_0)
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k, v, x: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

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



# Parsed testcases at query #5
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #6
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
    var_7 = (var_0, var_2)
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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
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
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True

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
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 10), (2, 30)])
    assert var_8 is True



# Parsed testcases at query #7
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True
    var_8 = lambda k, v, x: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(False)
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
    var_7 = 'a'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_7 = 'd'
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = callable(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



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
    var_6 = bool(var_5 == [('a', 1), ('b', 2)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'p'), (1, 'q'), (2, 'r')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.
# Failed to parse test_get_arity_with_mixed_args.




# Parsed testcases at query #11
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 'foo'), (1, 'bar')])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #12
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 11/13 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/11 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_2: var_3}
    var_8 = (var_0, var_7)
    var_9 = [var_8]
    var_10 = [var_2]

import pyrsistent._transformations as module_0

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
    var_9 = lambda x: x + var_2
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'b': 3})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_2: var_3}
    var_8 = (var_0, var_7)
    var_9 = [var_8]
    var_10 = [var_2]
    var_11 = lambda x: x + var_3
    var_12 = module_0._update_structure(var_6, var_9, var_10, var_11)
    var_13 = bool(var_12 == {'a': {'b': 2}, 'c': 2})
    assert var_13 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x + var_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = 'c'
    var_9 = [var_8]
    var_10 = lambda x: x
    var_11 = module_0._update_structure(var_4, var_7, var_9, var_10)
    var_12 = bool(var_11 == {'a': 1, 'b': 2})
    assert var_12 is True



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
    var_7 = False
    var_8 = lambda k: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #16
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k: k == var_0
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #20
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
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
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
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



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.
# Failed to parse test_get_arity_with_mixed_args.




# Parsed testcases at query #22
#--------------------------

# Failed to parse test__get_arity_with_default_parameters.




# Parsed testcases at query #23
#--------------------------

# Failed to parse test_arity_predicate_false.




# Parsed testcases at query #24
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = 'items'
    var_6 = 'method'
    var_7 = {var_5: var_6}
    var_8 = var_4 == var_7
    var_9 = bool(not var_8)
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test_\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'test_abc'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = '123_test'
    var_7 = var_1(var_6)
    assert var_7 is False
    var_8 = 'test_'
    var_9 = var_1(var_8)
    assert var_9 is False
    var_10 = 'test_123_extra'
    var_11 = var_1(var_10)
    assert var_11 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test_\\d+'
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
    var_8 = {}
    var_9 = var_1(var_8)
    assert var_9 is False



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_arity_with_default_args.




# Parsed testcases at query #28
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #29
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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True
    var_8 = lambda k, v, x: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(False)
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
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = bool(var_7 == [('b', 2)])
    assert var_8 is True

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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'd'
    var_8 = module_0._get_keys_and_values(var_6, var_7)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/9 statements.
# Partially parsed test__do_to_path_with_discard_command_and_non_existent_key. Retrieved 9/10 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x
    var_6 = []
    var_7 = module_0._do_to_path(var_4, var_6, var_5)
    var_8 = bool(var_7 == var_4)
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = []
    var_9 = module_0._do_to_path(var_4, var_8, var_7)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = lambda x: x * var_5
    var_8 = [var_0, var_2]
    var_9 = module_0._do_to_path(var_6, var_8, var_7)
    var_10 = bool(var_9 == {'a': {'b': 2}, 'c': 2})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = [var_0, var_2]
    var_9 = module_0._do_to_path(var_6, var_8, var_7)
    var_10 = bool(var_9 == {'a': {'b': 3}, 'c': 2})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = 'd'
    var_9 = [var_0, var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)
    var_11 = bool(var_10 == {'a': {'b': 1, 'd': 3}, 'c': 2})
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
    var_7 = lambda x: x * var_4
    var_8 = [var_0, var_1]
    var_9 = lambda k: k in var_8
    var_10 = [var_9]
    var_11 = module_0._do_to_path(var_6, var_10, var_7)
    var_12 = bool(var_11 == {'a': 2, 'b': 4, 'c': 3})
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
    var_7 = lambda x: x * var_4
    var_8 = lambda k, v: v > var_3
    var_9 = [var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)
    var_11 = bool(var_10 == {'a': 1, 'b': 4, 'c': 6})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'd'
    var_8 = [var_7]



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_single_positional_parameter.
# Failed to parse test_get_arity_with_multiple_positional_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.




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
    var_7 = (var_0, var_2)
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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
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
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #4
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k: k == var_0
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_callable_with_arity_greater_than_2. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #7
#--------------------------

# Failed to parse test__get_arity_with_all_parameters_having_defaults.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_from_object_with_existing_attribute. Retrieved 2/7 statements.
# Partially parsed test_get_from_object_with_non_existing_attribute. Retrieved 2/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._get(var_4, var_0, var_5)
    assert var_6 == 1

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = None
    var_7 = module_0._get(var_4, var_5, var_6)
    assert var_7 is None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0._get(var_3, var_0, var_4)
    assert var_5 == 2

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = None
    var_6 = module_0._get(var_3, var_4, var_5)
    assert var_6 is None

def test_case_0():
    var_0 = 'x'
    var_1 = None

def test_case_0():
    var_0 = 'y'
    var_1 = None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 'default'
    var_7 = module_0._get(var_4, var_5, var_6)
    assert var_7 == 'default'



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'a'
    var_12 = 1
    var_13 = (var_11, var_12)
    var_14 = bool(('a', 1) in var_9)
    assert var_14 is True
    var_15 = 'b'
    var_16 = 2
    var_17 = (var_15, var_16)
    var_18 = bool(('b', 2) in var_9)
    assert var_18 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #11
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 'foo'), (1, 'bar')])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #12
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda x, y, z: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test__get_arity_with_no_required_positional_args.




# Parsed testcases at query #16
#--------------------------

# Failed to parse test__get_arity_with_default_args.




# Parsed testcases at query #17
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test__get_arity_with_no_positional_args.




# Parsed testcases at query #19
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #21
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #22
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
    var_7 = module_0._items(var_6)
    var_8 = bool(var_7 == [('a', 1), ('b', 2), ('c', 3)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'p'), (1, 'q'), (2, 'r')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 10/14 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 13/28 statements.
# Partially parsed test__update_structure_with_empty_path_and_non_discard_command. Retrieved 12/15 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 14/28 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/13 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 8/14 statements.


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
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_3}
    var_9 = {var_5: var_6}
    var_10 = [var_2]
    var_11 = {}
    var_12 = {var_5: var_6}

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
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_3}
    var_9 = {var_5: var_6}
    var_10 = [var_2]
    var_11 = lambda x: x * var_6
    var_12 = {var_2: var_6}
    var_13 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x + var_2
    var_8 = 'c'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #24
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'b'
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
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
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



# Parsed testcases at query #25
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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #26
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #27
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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 13/15 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/11 statements.


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

import pyrsistent._transformations as module_0

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
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = bool(var_10 == {'a': 2, 'b': 4})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]

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
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]
    var_13 = lambda x: x * var_5
    var_14 = module_0._update_structure(var_8, var_11, var_12, var_13)
    var_15 = bool(var_14 == {'a': {'x': 2, 'y': 2}, 'b': 3})
    assert var_15 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x + var_2
    var_8 = 'c'



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('a', 1), ('b', 2)])
    assert var_10 is True



# Parsed testcases at query #31
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1)])
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
    var_7 = True
    var_8 = lambda k, v, x: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/12 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 8/18 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/13 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 9/19 statements.
# Partially parsed test__update_structure_with_no_changes. Retrieved 7/9 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_3}
    var_7 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = lambda x: x + var_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = lambda x: x + var_2
    var_7 = 2
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 14/19 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_discard_command. Retrieved 13/14 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_non_discard_command. Retrieved 16/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 8/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_empty_sentinel. Retrieved 15/26 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]
    var_13 = {var_3: var_5}

import pyrsistent._transformations as module_0

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
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)
    var_11 = 4
    var_12 = {var_0: var_3, var_1: var_11}

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
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = [var_10]
    var_12 = [var_2]
    var_13 = lambda x: x * var_5
    var_14 = module_0._update_structure(var_8, var_11, var_12, var_13)
    var_15 = {var_2: var_5, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = []
    var_7 = lambda x: x + var_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_2: var_4, var_3: var_5}
    var_10 = (var_0, var_9)
    var_11 = 'c'
    var_12 = 'z'
    var_13 = [var_12]
    var_14 = lambda x: x * var_5



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'evolver'
    var_2 = {}
    var_3 = []
    var_4 = 1
    var_5 = [var_4]
    var_6 = 'some_command'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/10 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 7/10 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 5/8 statements.


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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_2: var_3}
    var_8 = (var_0, var_7)
    var_9 = [var_8]
    var_10 = [var_2]
    var_11 = lambda x: x + var_3
    var_12 = module_0._update_structure(var_6, var_9, var_10, var_11)
    var_13 = bool(var_12 == {'a': {'b': 2}, 'c': 2})
    assert var_13 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 2
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x
    var_7 = module_0._update_structure(var_2, var_4, var_5, var_6)
    var_8 = bool(var_7 == {'a': 1})
    assert var_8 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #38
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
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
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True

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
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True



# Parsed testcases at query #39
#--------------------------

# Failed to parse test__get_arity_returns_false_for_predicate.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_items_with_non_dict_structure. Retrieved 5/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 1), (1, 2), (2, 3)])
    assert var_5 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = 'some'
    var_12 = 'path'
    var_13 = [var_11, var_12]
    var_14 = 'some_command'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 18/28 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = []
    var_8 = []
    var_9 = 'MockCommand'
    var_10 = ()
    var_11 = '__eq__'
    var_12 = 'discard'
    var_13 = lambda self, other: other.__name__ == var_12
    var_14 = {var_11: var_13}
    var_15 = [var_9, var_10, var_14]
    var_16 = 'MockDiscard'
    var_17 = ()
    var_18 = '__name__'
    var_19 = {var_18: var_12}
    var_20 = [var_16, var_17, var_19]



# Parsed testcases at query #44
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
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'some'
    var_6 = 'path'
    var_7 = [var_5, var_6]
    var_8 = 'some_command'



