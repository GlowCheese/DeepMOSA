####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_default.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_all_defaults.




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
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'missing'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = var_6[0][0]
    assert var_7 == 'missing'
    var_8 = var_6[0][1]

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 15
    var_6 = lambda idx, val: val > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(1, 20), (2, 30), (3, 40)])
    assert var_9 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'z'
    var_8 = lambda k: k == var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
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
    var_7 = 100
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/58 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 8/57 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 5/54 statements.
# Failed to parse test_do_to_path_with_single_key.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = []

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 42
    var_5 = []



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/26 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_evaluates_to_false. Retrieved 10/38 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'nonexistent'
    var_9 = lambda k: k == var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = 'not_a_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_items_with_dict.
# Failed to parse test_items_with_empty_dict.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
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
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True



# Parsed testcases at query #12
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
    var_11 = len(var_9)
    assert var_11 == 2
    var_12 = var_9[0][0]
    assert var_12 == 'a'
    var_13 = var_9[0][1]
    assert var_13 == 1
    var_14 = var_9[1][0]
    assert var_14 == 'c'
    var_15 = var_9[1][1]
    assert var_15 == 3



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/29 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_key'
    var_2 = callable(var_1)
    assert var_2 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



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
    var_7 = 'nonexistent'
    var_8 = lambda k: k == var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_string'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'no_match'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '12345'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = []
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+_[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc_123'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+_[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'ABC_123'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/25 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_update_structure_empty_path_with_discard. Retrieved 12/61 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 11/67 statements.
# Failed to parse test_update_structure_empty_sentinel_creates_pmap.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = (var_1, var_4)
    var_9 = (var_2, var_5)
    var_10 = [var_8, var_9]
    var_11 = []
    var_12 = {var_3: var_6}

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2: var_4, var_3: var_5}
    var_8 = [var_2]
    var_9 = 10
    var_10 = lambda x: var_9
    var_11 = {var_2: var_9, var_3: var_5}



# Parsed testcases at query #21
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = lambda k: k in var_6
    var_8 = module_0._get_keys_and_values(var_3, var_7)
    var_9 = bool(var_8 == [(0, 10), (2, 30)])
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
    var_4 = lambda k, v: v >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20), (2, 30)])
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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

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
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 2/5 statements.


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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'missing'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = var_6[0][0]
    assert var_7 == 'missing'
    var_8 = var_6[0][1]

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
    var_4 = module_0._get_keys_and_values(var_3, var_0)
    var_5 = var_4[0][0]
    assert var_5 == 10
    var_6 = var_4[0][1]

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(2, 30), (3, 40)])
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'nonexistent'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'zero'
    var_4 = 'one'
    var_5 = 'two'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = bool(var_7 == [(1, 'one')])
    assert var_8 is True

def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_predicate_at_line_1_evaluates_to_false.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_callable_key_spec_evaluates_to_true. Retrieved 22/48 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_13 is True
    var_14 = 'x'
    var_15 = 'y'
    var_16 = 'z'
    var_17 = 10
    var_18 = 20
    var_19 = 30
    var_20 = {var_14: var_17, var_15: var_18, var_16: var_19}
    var_21 = 15
    var_22 = lambda k, v: v > var_21
    var_23 = module_0._get_keys_and_values(var_20, var_22)
    var_24 = callable(var_22)
    var_25 = bool(var_24)
    assert var_25 is True
    var_26 = bool(var_23 == [('y', 20), ('z', 30)])
    assert var_26 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/19 statements.


def test_case_0():
    var_0 = 1



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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 1
    var_6 = lambda idx: idx > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [(2, 30), (3, 40)])
    assert var_8 is True

def test_case_0():
    pass



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_items_with_dict. Retrieved 7/9 statements.
# Failed to parse test_items_with_custom_dict_like_object.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {('a', 1), ('b', 2)})
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_get_keys_and_values_with_dict_and_non_callable_key. Retrieved 57/79 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = bool(var_8 == [('a', 1)])
    assert var_9 is True
    var_10 = 'd'
    var_11 = module_0._get_keys_and_values(var_7, var_10)
    var_12 = 10
    var_13 = 20
    var_14 = 30
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0._get_keys_and_values(var_15, var_4)
    var_17 = bool(var_16 == [(1, 20)])
    assert var_17 is True
    var_18 = [var_12, var_13, var_14]
    var_19 = 5
    var_20 = module_0._get_keys_and_values(var_18, var_19)
    var_21 = [var_1, var_3]
    var_22 = lambda k: k in var_21
    var_23 = module_0._get_keys_and_values(var_7, var_22)
    var_24 = sorted(var_23)
    var_25 = (var_1, var_4)
    var_26 = (var_3, var_6)
    var_27 = [var_25, var_26]
    var_28 = sorted(var_27)
    var_29 = bool(var_24 == var_28)
    assert var_29 is True
    var_30 = lambda k, v: v > var_4
    var_31 = module_0._get_keys_and_values(var_7, var_30)
    var_32 = sorted(var_31)
    var_33 = (var_2, var_5)
    var_34 = (var_3, var_6)
    var_35 = [var_33, var_34]
    var_36 = sorted(var_35)
    var_37 = bool(var_32 == var_36)
    assert var_37 is True
    var_38 = 100
    var_39 = 200
    var_40 = 300
    var_41 = [var_38, var_39, var_40]
    var_42 = 0
    var_43 = [var_42, var_5]
    var_44 = lambda idx: idx in var_43
    var_45 = module_0._get_keys_and_values(var_41, var_44)
    var_46 = sorted(var_45)
    var_47 = (var_42, var_38)
    var_48 = (var_5, var_40)
    var_49 = [var_47, var_48]
    var_50 = sorted(var_49)
    var_51 = bool(var_46 == var_50)
    assert var_51 is True
    var_52 = 150
    var_53 = lambda idx, val: val > var_52
    var_54 = module_0._get_keys_and_values(var_41, var_53)
    var_55 = sorted(var_54)
    var_56 = (var_4, var_39)
    var_57 = (var_5, var_40)
    var_58 = [var_56, var_57]
    var_59 = sorted(var_58)
    var_60 = bool(var_55 == var_59)
    assert var_60 is True
    var_61 = True
    var_62 = lambda a, b, c: var_61
    var_63 = module_0._get_keys_and_values(var_7, var_62)
    var_64 = bool(False)
    assert var_64 is True
    var_65 = 'callable in transform path must take 1 or 2 arguments'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_items_with_sequence_returns_enumerated_list. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = enumerate(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_all_default_parameters.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_mixed_parameters.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_keys_and_values_non_callable_predicate. Retrieved 10/31 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[0][0]
    assert var_10 == 'a'
    var_11 = var_8[0][1]
    assert var_11 == 1
    var_12 = callable(var_1)
    var_13 = bool(not var_12)
    assert var_13 is True



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
    var_7 = module_0._items(var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [('a', 1), ('b', 2), ('c', 3)])
    assert var_9 is True

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

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
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'nested'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0._items(var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [('key1', {'nested': 'value'}), ('key2', [1, 2])])
    assert var_11 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_keys_and_values_non_callable_with_object. Retrieved 1/6 statements.


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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_4 = 0
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = lambda k: k in var_6
    var_8 = module_0._get_keys_and_values(var_3, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = sorted(var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda : var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_0 = 'x'



# Parsed testcases at query #34
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_4 = 0
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = lambda k: k in var_6
    var_8 = module_0._get_keys_and_values(var_3, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 20), (2, 30)])
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 7/56 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 5/54 statements.
# Failed to parse test_do_to_path_with_single_key_path.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = lambda x: x * var_4
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 42
    var_5 = []



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 12/40 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'a'
    var_13 = 1
    var_14 = (var_12, var_13)
    var_15 = bool(('a', 1) in var_10)
    assert var_15 is True
    var_16 = 'c'
    var_17 = 3
    var_18 = (var_16, var_17)
    var_19 = bool(('c', 3) in var_10)
    assert var_19 is True
    var_20 = callable(var_9)
    assert var_20 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_path_not_empty. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 'simple_key'
    var_2 = callable(var_1)
    assert var_2 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 23/53 statements.


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
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True
    var_12 = [var_0, var_2]
    var_13 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_13 is True
    var_14 = 'x'
    var_15 = 'y'
    var_16 = 'z'
    var_17 = 10
    var_18 = 20
    var_19 = 30
    var_20 = {var_14: var_17, var_15: var_18, var_16: var_19}
    var_21 = 15
    var_22 = lambda k, v: v > var_21
    var_23 = module_0._get_keys_and_values(var_20, var_22)
    var_24 = len(var_23)
    var_25 = bool(var_24 > 0)
    assert var_25 is True
    var_26 = bool(var_23 == [('y', 20), ('z', 30)])
    assert var_26 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 22/45 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True
    var_12 = callable(var_9)
    assert var_12 is True
    var_13 = 'x'
    var_14 = 'y'
    var_15 = 'z'
    var_16 = 10
    var_17 = 20
    var_18 = 30
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = 15
    var_21 = lambda k, v: v > var_20
    var_22 = module_0._get_keys_and_values(var_19, var_21)
    var_23 = bool(var_22 == [('y', 20), ('z', 30)])
    assert var_23 is True
    var_24 = callable(var_21)
    assert var_24 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 10/61 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 7/56 statements.
# Failed to parse test_do_to_path_with_single_key_path.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = []
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_7: var_8}

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = []



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 12/35 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = len(var_10)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 1
    var_15 = (var_13, var_14)
    var_16 = bool(('a', 1) in var_10)
    assert var_16 is True
    var_17 = 'c'
    var_18 = 3
    var_19 = (var_17, var_18)
    var_20 = bool(('c', 3) in var_10)
    assert var_20 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_at_line_1_evaluates_to_false. Retrieved 11/39 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0._get_keys_and_values(var_9, var_2)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = callable(var_2)
    assert var_12 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 14/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = None
    var_7 = lambda x, y: var_6
    var_8 = 'some'
    var_9 = 'path'
    var_10 = [var_8, var_9]
    var_11 = 'key'
    var_12 = [var_11]
    var_13 = lambda x, y: var_6



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_non_empty_path. Retrieved 14/25 statements.
# Partially parsed test_update_structure_with_missing_value_and_discard. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_missing_value_creates_pmap. Retrieved 6/14 statements.
# Partially parsed test_update_structure_multiple_kvs_with_callable_command. Retrieved 13/17 statements.
# Partially parsed test_update_structure_discard_multiple_keys_reversed. Retrieved 15/20 statements.
# Partially parsed test_update_structure_nested_path_with_command. Retrieved 11/28 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = {var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_2: var_4, var_3: var_5}
    var_9 = [var_2]
    var_10 = 100
    var_11 = lambda x: x + var_10
    var_12 = 101
    var_13 = {var_2: var_12, var_3: var_5}

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
    var_5 = lambda x: x

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = 2
    var_10 = lambda x: x * var_9
    var_11 = 20
    var_12 = {var_0: var_3, var_1: var_11}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = (var_3, var_7)
    var_10 = (var_1, var_5)
    var_11 = (var_0, var_4)
    var_12 = [var_9, var_10, var_11]
    var_13 = []
    var_14 = {var_2: var_6}

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = [var_1, var_2]
    var_7 = 8
    var_8 = lambda x: x + var_7
    var_9 = 50
    var_10 = {var_2: var_9}



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = (var_0, var_1)
    var_5 = [var_4]
    var_6 = 'some'
    var_7 = 'path'
    var_8 = [var_6, var_7]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 11/19 statements.
# Partially parsed test_update_structure_creates_empty_pmap_when_value_missing. Retrieved 6/14 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_discard. Retrieved 6/12 statements.
# Partially parsed test_update_structure_preserves_unchanged_values. Retrieved 12/15 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 12/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = {var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = (var_1,)
    var_8 = [var_7]
    var_9 = 10
    var_10 = lambda x: var_9

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_1,)
    var_3 = [var_2]
    var_4 = 42
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = [var_7]
    var_9 = []
    var_10 = 100
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_2, var_5)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = lambda x: x * var_4



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_true. Retrieved 9/16 statements.


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
    var_9 = 'a'
    var_10 = 'b'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_false. Retrieved 13/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = 20
    var_8 = (var_1, var_7)
    var_9 = [var_6, var_8]
    var_10 = 'some'
    var_11 = 'path'
    var_12 = [var_10, var_11]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 9/14 statements.


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



# Parsed testcases at query #51
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_all_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_defaults.
# Failed to parse test_get_arity_ignores_var_args.
# Failed to parse test_get_arity_ignores_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #52
#--------------------------

# Partially parsed test_items_with_dict. Retrieved 7/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {('a', 1), ('b', 2)})
    assert var_7 is True

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_5 is True

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
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0._items(var_5)
    var_7 = list(var_6)
    var_8 = 'key'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = (var_8, var_12)
    var_14 = bool(('key', [1, 2, 3]) in var_7)
    assert var_14 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_callable_predicate_evaluates_to_false. Retrieved 7/35 statements.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 12/35 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = len(var_10)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 1
    var_15 = (var_13, var_14)
    var_16 = bool(('a', 1) in var_10)
    assert var_16 is True
    var_17 = 'c'
    var_18 = 3
    var_19 = (var_17, var_18)
    var_20 = bool(('c', 3) in var_10)
    assert var_20 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 9/14 statements.


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
    var_9 = bool(not var_8)
    assert var_9 is True



# Parsed testcases at query #56
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0][0]
    assert var_7 == 5

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 15
    var_6 = lambda idx, val: val > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(1, 20), (2, 30), (3, 40)])
    assert var_9 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_2 = lambda idx: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True



# Parsed testcases at query #57
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
    var_7 = 'a'
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = callable(var_7)
    assert var_9 is False
    var_10 = bool(var_8 == [('a', 1)])
    assert var_10 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 15/30 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = 20
    var_8 = (var_1, var_7)
    var_9 = [var_6, var_8]
    var_10 = 'some'
    var_11 = 'path'
    var_12 = [var_10, var_11]
    var_13 = 'discard'
    var_14 = False



# Parsed testcases at query #59
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_all_default_parameters.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #60
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/37 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True
    var_12 = callable(var_9)
    assert var_12 is True
    var_13 = lambda k, v: v > var_4
    var_14 = module_0._get_keys_and_values(var_7, var_13)
    var_15 = bool(var_14 == [('b', 2), ('c', 3)])
    assert var_15 is True
    var_16 = callable(var_13)
    assert var_16 is True



# Parsed testcases at query #61
#--------------------------




def test_case_0():
    var_0 = 'not_a_function'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_get_keys_and_values_non_callable_predicate. Retrieved 10/32 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'a'
    var_9 = module_0._get_keys_and_values(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = var_9[0][0]
    assert var_11 == 'a'
    var_12 = var_9[0][1]
    assert var_12 == 1



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_items_with_ordered_dict. Retrieved 7/11 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = (var_7, var_8)
    var_10 = bool(('a', 1) in var_6)
    assert var_10 is True
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = bool(('b', 2) in var_6)
    assert var_14 is True
    var_15 = len(var_6)
    assert var_15 == 2

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 10), (1, 20), (2, 30)])
    assert var_5 is True

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
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True



# Parsed testcases at query #64
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
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'missing'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'missing'

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
    var_4 = module_0._get_keys_and_values(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 10

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val > var_1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(2, 15), (3, 20)])
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = 20
    var_8 = (var_1, var_7)
    var_9 = [var_6, var_8]
    var_10 = 'some'
    var_11 = 'path'
    var_12 = [var_10, var_11]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 2/7 statements.
# Partially parsed test_do_to_path_single_key_in_dict. Retrieved 8/11 statements.
# Partially parsed test_do_to_path_nested_path_in_dict. Retrieved 7/12 statements.
# Partially parsed test_do_to_path_with_list_index. Retrieved 7/10 statements.
# Partially parsed test_do_to_path_with_unary_predicate. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 11/14 statements.
# Partially parsed test_do_to_path_discard_with_empty_path. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_discard_key_in_dict. Retrieved 8/11 statements.
# Partially parsed test_do_to_path_missing_key_creates_empty_pmap. Retrieved 8/11 statements.
# Partially parsed test_do_to_path_deeply_nested. Retrieved 8/15 statements.
# Partially parsed test_do_to_path_invalid_predicate_arity. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 5
    var_1 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = 10
    var_3 = module_0._do_to_path(var_0, var_1, var_2)
    assert var_3 == 10

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 10
    var_7 = lambda x: x + var_6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 2
    var_6 = lambda x: x * var_5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0]
    var_5 = 10
    var_6 = lambda x: x + var_5

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
    var_9 = [var_8]
    var_10 = 10
    var_11 = lambda x: x + var_10

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
    var_10 = lambda x: x + var_9

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_1]
    var_8 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = lambda x: var_6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_2]
    var_6 = 3
    var_7 = lambda x: x * var_6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda x, y, z: var_5
    var_7 = [var_6]
    var_8 = lambda x: x
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'callable in transform path must take 1 or 2 arguments'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_with_object_attribute. Retrieved 3/6 statements.
# Partially parsed test_get_with_object_missing_attribute. Retrieved 2/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default'
    var_6 = module_0._get(var_4, var_0, var_5)
    assert var_6 == 1

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'default'
    var_5 = module_0._get(var_2, var_3, var_4)
    assert var_5 == 'default'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 'default'
    var_6 = module_0._get(var_3, var_4, var_5)
    assert var_6 == 20

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'default'
    var_5 = module_0._get(var_2, var_3, var_4)
    assert var_5 == 'default'

def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'
    var_2 = 'default'

def test_case_0():
    var_0 = 'missing'
    var_1 = 'default'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = (var_0, var_1, var_2)
    var_4 = 2
    var_5 = 'default'
    var_6 = module_0._get(var_3, var_4, var_5)
    assert var_6 == 300

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 10
    var_4 = 'default'
    var_5 = module_0._get(var_2, var_3, var_4)
    assert var_5 == 'default'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'any_key'
    var_2 = 'default_value'
    var_3 = module_0._get(var_0, var_1, var_2)
    assert var_3 == 'default_value'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'default_value'
    var_3 = module_0._get(var_0, var_1, var_2)
    assert var_3 == 'default_value'



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
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'missing'
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
    var_4 = module_0._get_keys_and_values(var_3, var_0)

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
    var_10 = set(var_9)
    var_11 = bool(var_10 == {('a', 1), ('c', 3)})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda i: i % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = set(var_8)
    var_10 = bool(var_9 == {(0, 10), (2, 30)})
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
    var_9 = set(var_8)
    var_10 = bool(var_9 == {('b', 2), ('c', 3)})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda i, v: v >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = set(var_6)
    var_8 = bool(var_7 == {(2, 30), (3, 40)})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k, v, extra: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'key'
    var_2 = module_0._get_keys_and_values(var_0, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._get_keys_and_values(var_0, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 100
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_items_with_dict. Retrieved 7/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {('a', 1), ('b', 2)})
    assert var_7 is True

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
    var_0 = 100
    var_1 = 200
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 100), (1, 200)])
    assert var_4 is True

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
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = module_0._items(var_6)
    var_8 = bool(var_7 == [(0, {'x': 1}), (1, {'y': 2})])
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_items_with_custom_dict_like.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = set(var_6)
    var_8 = bool(var_7 == {('a', 1), ('b', 2)})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 'x'), (1, 'y'), (2, 'z')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 'a'), (1, 'b'), (2, 'c')])
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
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [(0, 0), (1, 1), (2, 2)])
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_value'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'no_match'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc_123'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'ABC_123'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = var_1(var_3)
    assert var_4 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_suffix'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\.\\w+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test.abc'
    var_3 = var_1(var_2)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_defaults.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_do_to_path_with_callable_command. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'
    var_2 = 'kwargs'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_evaluates_to_false. Retrieved 13/34 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'nonexistent'
    var_9 = lambda k: k == var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = callable(var_9)
    assert var_12 is True
    var_13 = var_9(var_1)
    assert var_13 is False
    var_14 = var_9(var_8)
    assert var_14 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_defaults.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_callable_command. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_structure'
    var_1 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/23 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 8/36 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'my_key'
    var_2 = 'my_key'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = 'other_value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = bool(var_7 == [('my_key', 'value')])
    assert var_8 is True
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 22/45 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_12 is True
    var_13 = 'x'
    var_14 = 'y'
    var_15 = 'z'
    var_16 = 10
    var_17 = 20
    var_18 = 30
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = 15
    var_21 = lambda k, v: v > var_20
    var_22 = module_0._get_keys_and_values(var_19, var_21)
    var_23 = callable(var_21)
    assert var_23 is True
    var_24 = bool(var_22 == [('y', 20), ('z', 30)])
    assert var_24 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_callable_command. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 8/20 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 11/16 statements.
# Partially parsed test_update_structure_with_vector. Retrieved 9/14 statements.
# Partially parsed test_update_structure_discard_with_reversed_kvs. Retrieved 8/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_expansion. Retrieved 8/15 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 42

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 99
    var_7 = {var_1: var_6}

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
    var_9 = 10
    var_10 = {var_0: var_9, var_1: var_9}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = (var_0, var_1)
    var_6 = [var_4, var_5]
    var_7 = []
    var_8 = 99

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = (var_2, var_3)
    var_5 = (var_0, var_1)
    var_6 = [var_4, var_5]
    var_7 = []

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
    var_5 = 42
    var_6 = lambda x: var_5
    var_7 = {var_0: var_1, var_3: var_5}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key. Retrieved 8/29 statements.
# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 10/31 statements.
# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 9/30 statements.
# Partially parsed test_get_keys_and_values_with_list. Retrieved 6/27 statements.
# Failed to parse test_get_keys_and_values_with_invalid_arity.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = bool(var_8 == [('a', 1)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = lambda k, v: v > var_4
    var_9 = module_0._get_keys_and_values(var_7, var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 10), (1, 20), (2, 30)])
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




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
    var_7 = module_0._get_keys_and_values(var_6, var_0)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'missing'
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
    var_4 = module_0._get_keys_and_values(var_3, var_0)

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda i: i % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda i, v: v >= var_1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 20), (2, 30), (3, 40)])
    assert var_8 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'z'
    var_8 = lambda k: k == var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
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
    var_7 = 100
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_callable_command. Retrieved 2/8 statements.
# Partially parsed test_predicate_non_callable_command. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'not_callable'
    var_1 = 42
    var_2 = []
    var_3 = callable(var_0)
    var_4 = var_0(var_1)
    var_5 = callable(var_0)
    assert var_5 is False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key. Retrieved 8/29 statements.
# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 11/32 statements.
# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 10/31 statements.
# Partially parsed test_get_keys_and_values_with_list_and_non_callable. Retrieved 6/27 statements.
# Failed to parse test_get_keys_and_values_with_invalid_arity.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = bool(var_8 == [('a', 1)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = (var_1, var_3)
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'a'
    var_13 = 1
    var_14 = (var_12, var_13)
    var_15 = bool(('a', 1) in var_10)
    assert var_15 is True
    var_16 = 'c'
    var_17 = 3
    var_18 = (var_16, var_17)
    var_19 = bool(('c', 3) in var_10)
    assert var_19 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = lambda k, v: v > var_4
    var_9 = module_0._get_keys_and_values(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = bool(('b', 2) in var_9)
    assert var_14 is True
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = bool(('c', 3) in var_9)
    assert var_18 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_callable_command. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_items_with_ordered_dict. Retrieved 7/11 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = (var_7, var_8)
    var_10 = bool(('a', 1) in var_6)
    assert var_10 is True
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = bool(('b', 2) in var_6)
    assert var_14 is True
    var_15 = len(var_6)
    assert var_15 == 2

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

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
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 5), (1, 15), (2, 25)])
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
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._items(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 0
    var_7 = [r[var_6] for r in var_4]
    var_8 = [r[var_0] for r in var_4]
    var_9 = set(var_7)
    var_10 = bool(var_9 == {0, 1, 2})
    assert var_10 is True
    var_11 = set(var_8)
    var_12 = bool(var_11 == {1, 2, 3})
    assert var_12 is True



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 10/24 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = module_0._items(var_6)
    var_9 = [(k, v) for (k, v) in var_8 if key_spec(k)]
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_predicate_at_line_5_evaluates_to_false.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_true. Retrieved 12/39 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_2]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = len(var_10)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 1
    var_15 = (var_13, var_14)
    var_16 = bool(('a', 1) in var_10)
    assert var_16 is True
    var_17 = 'b'
    var_18 = 2
    var_19 = (var_17, var_18)
    var_20 = bool(('b', 2) in var_10)
    assert var_20 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 9/18 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_discard. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_expansion. Retrieved 6/13 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 12/16 statements.
# Partially parsed test_update_structure_reversed_kvs_with_discard. Retrieved 12/17 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = True
    var_6 = lambda k, v: var_5
    var_7 = [var_6]
    var_8 = lambda x: x + var_5
    var_9 = 'a'

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
    var_5 = lambda x: x
    var_6 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = lambda x: x * var_4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = (var_1, var_2)
    var_7 = (var_0, var_1)
    var_8 = 0
    var_9 = (var_8, var_0)
    var_10 = [var_6, var_7, var_9]
    var_11 = []



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attributes. Retrieved 1/7 statements.


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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_4 = 0
    var_5 = lambda k: k > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 20), (2, 30)])
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
    var_4 = lambda k, v: v >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = sorted(var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_0 = 'x'



# Parsed testcases at query #35
#--------------------------




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
    var_7 = var_6[0][0]
    assert var_7 == 'c'
    var_8 = var_6[0][1]

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda i: i % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda i, v: v >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = sorted(var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 200)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 5
    var_6 = var_4[0][1]



# Parsed testcases at query #36
#--------------------------




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
    var_7 = var_6[0][0]
    assert var_7 == 'c'
    var_8 = var_6[0][1].__class__.__name__
    assert var_8 == 'object'

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda k: k % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = bool(var_8 == [(0, 10), (2, 30)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 15
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [(1, 20), (2, 30), (3, 40)])
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

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
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #37
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_10 = set(var_9)
    var_11 = bool(var_10 == {('a', 1), ('c', 3)})
    assert var_11 is True

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
    var_9 = set(var_8)
    var_10 = bool(var_9 == {(0, 10), (2, 30)})
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
    var_9 = set(var_8)
    var_10 = bool(var_9 == {('b', 2), ('c', 3)})
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
    var_7 = set(var_6)
    var_8 = bool(var_7 == {(1, 20), (2, 30)})
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

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
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 9/17 statements.


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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 23/51 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True
    var_12 = len(var_10)
    var_13 = bool(var_12 > 0)
    assert var_13 is True
    var_14 = lambda k, v: v > var_4
    var_15 = module_0._get_keys_and_values(var_7, var_14)
    var_16 = bool(var_15 == [('b', 2), ('c', 3)])
    assert var_16 is True
    var_17 = len(var_15)
    var_18 = bool(var_17 > 0)
    assert var_18 is True
    var_19 = 10
    var_20 = 20
    var_21 = 30
    var_22 = 40
    var_23 = [var_19, var_20, var_21, var_22]
    var_24 = 0
    var_25 = lambda i: i % var_5 == var_24
    var_26 = module_0._get_keys_and_values(var_23, var_25)
    var_27 = bool(var_26 == [(0, 10), (2, 30)])
    assert var_27 is True
    var_28 = len(var_26)
    var_29 = bool(var_28 > 0)
    assert var_29 is True



# Parsed testcases at query #41
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda k: k > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(0, 10), (1, 20), (2, 30)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
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
    var_9 = bool(var_8 == [('b', 2), ('c', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda k, v: v % var_1 == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 10)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda a, b, c: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_4 = module_0._get_keys_and_values(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 10



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_callable_predicate_with_arity_1. Retrieved 10/38 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/61 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 8/61 statements.
# Failed to parse test_update_structure_with_transform_command.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = (var_1, var_4)
    var_9 = (var_2, var_5)
    var_10 = [var_8, var_9]
    var_11 = []
    var_12 = {var_3: var_6}

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'nested'
    var_7 = [var_1]
    var_8 = {var_2: var_4}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 24/47 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = len(var_10)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 1
    var_15 = (var_13, var_14)
    var_16 = bool(('a', 1) in var_10)
    assert var_16 is True
    var_17 = 'c'
    var_18 = 3
    var_19 = (var_17, var_18)
    var_20 = bool(('c', 3) in var_10)
    assert var_20 is True
    var_21 = 'x'
    var_22 = 'y'
    var_23 = 'z'
    var_24 = 10
    var_25 = 20
    var_26 = 30
    var_27 = {var_21: var_24, var_22: var_25, var_23: var_26}
    var_28 = 15
    var_29 = lambda k, v: v > var_28
    var_30 = module_0._get_keys_and_values(var_27, var_29)
    var_31 = callable(var_29)
    assert var_31 is True
    var_32 = len(var_30)
    assert var_32 == 2
    var_33 = 'y'
    var_34 = 20
    var_35 = (var_33, var_34)
    var_36 = bool(('y', 20) in var_30)
    assert var_36 is True
    var_37 = 'z'
    var_38 = 30
    var_39 = (var_37, var_38)
    var_40 = bool(('z', 30) in var_30)
    assert var_40 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 12/61 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 10/66 statements.
# Failed to parse test_update_structure_with_command_callable.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = (var_1, var_4)
    var_9 = (var_2, var_5)
    var_10 = [var_8, var_9]
    var_11 = []
    var_12 = {var_3: var_6}

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = [var_2]
    var_7 = 5
    var_8 = lambda x: x + var_7
    var_9 = 15
    var_10 = {var_2: var_9}



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_empty_path_and_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_path_with_callable. Retrieved 13/24 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 16/20 statements.
# Partially parsed test_update_structure_discard_multiple_kvs_reversed. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_creates_pmap. Retrieved 6/15 statements.
# Partially parsed test_update_structure_nested_with_empty_sentinel. Retrieved 9/18 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 42

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_2: var_4, var_3: var_5}
    var_9 = lambda k, v: k == var_2
    var_10 = [var_9]
    var_11 = lambda x: x * var_5
    var_12 = {var_2: var_5, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = 10
    var_12 = lambda x: x + var_11
    var_13 = 11
    var_14 = 12
    var_15 = {var_0: var_13, var_1: var_14, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = {var_2: var_5}

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
    var_5 = lambda x: x

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = 5
    var_7 = 5
    var_8 = {var_4: var_7}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = []



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 12/23 statements.
# Partially parsed test_update_structure_discard_multiple_kvs. Retrieved 13/17 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard. Retrieved 7/16 statements.
# Partially parsed test_update_structure_no_change_when_result_equals_original. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_vector. Retrieved 14/18 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = 5
    var_9 = lambda v: v + var_8
    var_10 = 6
    var_11 = {var_1: var_10, var_2: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_2, var_5)
    var_8 = (var_1, var_4)
    var_9 = (var_0, var_3)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = lambda x: x
    var_6 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = lambda x: x
    var_9 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = (var_4, var_0)
    var_6 = (var_0, var_1)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = 10
    var_10 = lambda x: x + var_9
    var_11 = 11
    var_12 = 12
    var_13 = [var_11, var_12, var_2]



# Parsed testcases at query #49
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_all_default_parameters.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_mixed_parameters.




# Parsed testcases at query #51
#--------------------------

# Failed to parse test_predicate_callable_check_line_1.




# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'string_key'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #54
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = (var_0, var_1)
    var_8 = (var_1, var_2)
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(var_4 == var_9)
    assert var_10 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/42 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_2]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'a'
    var_13 = 1
    var_14 = (var_12, var_13)
    var_15 = bool(('a', 1) in var_10)
    assert var_15 is True
    var_16 = 'b'
    var_17 = 2
    var_18 = (var_16, var_17)
    var_19 = bool(('b', 2) in var_10)
    assert var_19 is True
    var_20 = lambda k, v: v > var_4
    var_21 = module_0._get_keys_and_values(var_7, var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'b'
    var_24 = 2
    var_25 = (var_23, var_24)
    var_26 = bool(('b', 2) in var_21)
    assert var_26 is True
    var_27 = 'c'
    var_28 = 3
    var_29 = (var_27, var_28)
    var_30 = bool(('c', 3) in var_21)
    assert var_30 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = (var_0, var_7)
    var_9 = (var_1, var_7)
    var_10 = [var_8, var_9]
    var_11 = []



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = None
    var_5 = lambda x, y: var_4
    var_6 = var_5 is var_4
    var_7 = 'some'
    var_8 = 'path'
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = 'not_discard'



# Parsed testcases at query #59
#--------------------------




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
    var_12 = bool(var_4 == var_11)
    assert var_12 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_get_keys_and_values_with_dict_and_literal_key. Retrieved 6/27 statements.
# Partially parsed test_get_keys_and_values_with_dict_and_unary_predicate. Retrieved 10/31 statements.
# Partially parsed test_get_keys_and_values_with_dict_and_binary_predicate. Retrieved 9/30 statements.
# Partially parsed test_get_keys_and_values_with_list_and_literal_index. Retrieved 6/27 statements.
# Failed to parse test_get_keys_and_values_with_invalid_arity.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = lambda k, v: v > var_4
    var_9 = module_0._get_keys_and_values(var_7, var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #62
#--------------------------

# Partially parsed test_get_arity. Retrieved 4/34 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 1
    var_2 = lambda x, y: x + y
    var_3 = module_0._get_arity(var_2)
    assert var_3 == 2



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_get_keys_and_values_with_dict_and_string_key. Retrieved 38/60 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True
    var_8 = {var_1: var_3, var_2: var_4}
    var_9 = 'c'
    var_10 = module_0._get_keys_and_values(var_8, var_9)
    var_11 = var_10[0][0]
    assert var_11 == 'c'
    var_12 = var_10[0][1]
    var_13 = 10
    var_14 = 20
    var_15 = 30
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0._get_keys_and_values(var_16, var_3)
    var_18 = bool(var_17 == [(1, 20)])
    assert var_18 is True
    var_19 = 3
    var_20 = {var_1: var_3, var_2: var_4, var_9: var_19}
    var_21 = [var_1, var_9]
    var_22 = lambda k: k in var_21
    var_23 = module_0._get_keys_and_values(var_20, var_22)
    var_24 = sorted(var_23)
    var_25 = bool(var_24 == [('a', 1), ('c', 3)])
    assert var_25 is True
    var_26 = {var_1: var_3, var_2: var_4, var_9: var_19}
    var_27 = lambda k, v: v > var_3
    var_28 = module_0._get_keys_and_values(var_26, var_27)
    var_29 = sorted(var_28)
    var_30 = bool(var_29 == [('b', 2), ('c', 3)])
    assert var_30 is True
    var_31 = [var_13, var_14, var_15]
    var_32 = 0
    var_33 = lambda i: i % var_4 == var_32
    var_34 = module_0._get_keys_and_values(var_31, var_33)
    var_35 = bool(var_34 == [(0, 10), (2, 30)])
    assert var_35 is True
    var_36 = [var_13, var_14, var_15]
    var_37 = 15
    var_38 = lambda i, v: v > var_37
    var_39 = module_0._get_keys_and_values(var_36, var_38)
    var_40 = bool(var_39 == [(1, 20), (2, 30)])
    assert var_40 is True
    var_41 = 'a'
    var_42 = 1
    var_43 = {var_41: var_42}
    var_44 = True
    var_45 = lambda x, y, z: var_44
    var_46 = module_0._get_keys_and_values(var_43, var_45)
    var_47 = bool(False)
    assert var_47 is True
    var_48 = 'callable in transform path must take 1 or 2 arguments'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/37 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True
    var_12 = callable(var_9)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = lambda k, v: v > var_4
    var_15 = module_0._get_keys_and_values(var_7, var_14)
    var_16 = bool(var_15 == [('b', 2), ('c', 3)])
    assert var_16 is True
    var_17 = callable(var_14)
    var_18 = bool(var_17)
    assert var_18 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key. Retrieved 39/61 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = bool(var_6 == [('a', 1)])
    assert var_7 is True
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0._get_keys_and_values(var_11, var_3)
    var_13 = bool(var_12 == [(1, 20)])
    assert var_13 is True
    var_14 = {var_1: var_3}
    var_15 = 'c'
    var_16 = module_0._get_keys_and_values(var_14, var_15)
    var_17 = var_16[0][0]
    assert var_17 == 'c'
    var_18 = var_16[0][1]
    var_19 = 3
    var_20 = {var_1: var_3, var_2: var_4, var_15: var_19}
    var_21 = [var_1, var_15]
    var_22 = lambda k: k in var_21
    var_23 = module_0._get_keys_and_values(var_20, var_22)
    var_24 = sorted(var_23)
    var_25 = bool(var_24 == [('a', 1), ('c', 3)])
    assert var_25 is True
    var_26 = {var_1: var_3, var_2: var_4, var_15: var_19}
    var_27 = lambda k, v: v > var_3
    var_28 = module_0._get_keys_and_values(var_26, var_27)
    var_29 = sorted(var_28)
    var_30 = bool(var_29 == [('b', 2), ('c', 3)])
    assert var_30 is True
    var_31 = 40
    var_32 = [var_8, var_9, var_10, var_31]
    var_33 = 0
    var_34 = lambda idx: idx % var_4 == var_33
    var_35 = module_0._get_keys_and_values(var_32, var_34)
    var_36 = bool(var_35 == [(0, 10), (2, 30)])
    assert var_36 is True
    var_37 = [var_8, var_9, var_10, var_31]
    var_38 = 15
    var_39 = lambda idx, val: val > var_38
    var_40 = module_0._get_keys_and_values(var_37, var_39)
    var_41 = bool(var_40 == [(1, 20), (2, 30), (3, 40)])
    assert var_41 is True
    var_42 = 'a'
    var_43 = 1
    var_44 = {var_42: var_43}
    var_45 = True
    var_46 = lambda x, y, z: var_45
    var_47 = module_0._get_keys_and_values(var_44, var_46)
    var_48 = bool(False)
    assert var_48 is True
    var_49 = 'callable in transform path must take 1 or 2 arguments'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 20/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = None
    var_10 = lambda e, k: var_9
    var_11 = bool(not (not var_8 and var_10 is None))
    assert var_11 is True
    var_12 = (var_0, var_5)
    var_13 = [var_12]
    var_14 = 'x'
    var_15 = [var_14]
    var_16 = None
    var_17 = bool(not (not var_15 and var_16 is None))
    assert var_17 is True
    var_18 = (var_0, var_5)
    var_19 = [var_18]
    var_20 = [var_14]
    var_21 = lambda e, k: var_9
    var_22 = bool(not (not var_20 and var_21 is None))
    assert var_22 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 8/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = bool(not var_0)
    assert var_8 is True



# Parsed testcases at query #68
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #69
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(2, 30), (3, 40)])
    assert var_8 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda : var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #70
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_default.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_with_keyword_only_parameters.




# Parsed testcases at query #71
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_is_false. Retrieved 15/35 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = [var_4, var_5]
    var_7 = (var_0, var_1)
    var_8 = [var_7]
    var_9 = []
    var_10 = None
    var_11 = lambda e, k: var_10
    var_12 = []
    var_13 = []
    var_14 = 'some_other_command'



# Parsed testcases at query #72
#--------------------------




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
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'missing'

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
    var_10 = set(var_9)
    var_11 = bool(var_10 == {('a', 1), ('c', 3)})
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = set(var_8)
    var_10 = bool(var_9 == {(0, 10), (2, 30)})
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
    var_9 = set(var_8)
    var_10 = bool(var_9 == {('b', 2), ('c', 3)})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val >= var_1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = set(var_6)
    var_8 = bool(var_7 == {(1, 10), (2, 15), (3, 20)})
    assert var_8 is True

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
    var_7 = 'callable in transform path must take 1 or 2 arguments'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'z'
    var_6 = lambda k: k == var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 100
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #73
#--------------------------




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
    var_12 = bool(var_4 == var_11)
    assert var_12 is True



# Parsed testcases at query #74
#--------------------------




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
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'c'

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
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0][0]
    assert var_7 == 5

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
    var_10 = sorted(var_9)
    var_11 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = 0
    var_7 = lambda i: i % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
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
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [('b', 2), ('c', 3)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 25
    var_6 = lambda i, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(2, 30), (3, 40)])
    assert var_9 is True

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
    var_9 = 'callable in transform path must take 1 or 2 arguments'

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
    var_2 = lambda i, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




