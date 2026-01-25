####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 8/14 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_single_key_path. Retrieved 9/19 statements.
# Partially parsed test_do_to_path_nested_path. Retrieved 10/24 statements.
# Partially parsed test_do_to_path_with_callable_predicate_unary. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_callable_predicate_binary. Retrieved 11/14 statements.
# Partially parsed test_do_to_path_with_list_structure. Retrieved 7/10 statements.
# Partially parsed test_do_to_path_missing_key_creates_empty_pmap. Retrieved 7/19 statements.
# Partially parsed test_do_to_path_with_discard_command. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = {var_4: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'replacement'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0]
    var_5 = 'c'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1]
    var_6 = 'd'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}

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
    var_11 = lambda x: x * var_10

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0]
    var_5 = 20
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 'a'



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0]
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_optional.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_keyword_only.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_items_with_ordered_dict. Retrieved 7/11 statements.
# Failed to parse test_items_with_custom_object_with_items_method.


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
    var_9 = 'a'
    var_10 = 1
    var_11 = (var_9, var_10)
    var_12 = bool(('a', 1) in var_8)
    assert var_12 is True
    var_13 = 'b'
    var_14 = 2
    var_15 = (var_13, var_14)
    var_16 = bool(('b', 2) in var_8)
    assert var_16 is True
    var_17 = 'c'
    var_18 = 3
    var_19 = (var_17, var_18)
    var_20 = bool(('c', 3) in var_8)
    assert var_20 is True
    var_21 = len(var_8)
    assert var_21 == 3

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
    var_5 = list(var_4)
    var_6 = bool(var_5 == [(0, 10), (1, 20), (2, 30)])
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
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_3 is True

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



# Parsed testcases at query #5
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'hello_world'
    var_5 = var_1(var_4)
    assert var_5 is True
    var_6 = 'world_hello'
    var_7 = var_1(var_6)
    assert var_7 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = '456'
    var_5 = var_1(var_4)
    assert var_5 is True
    var_6 = '123abc'
    var_7 = var_1(var_6)
    assert var_7 is False
    var_8 = 'abc123'
    var_9 = var_1(var_8)
    assert var_9 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'a'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+\\.'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc.'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'xyz.def'
    var_5 = var_1(var_4)
    assert var_5 is True
    var_6 = 'ABC.'
    var_7 = var_1(var_6)
    assert var_7 is False
    var_8 = 'abc'
    var_9 = var_1(var_8)
    assert var_9 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^start'
    var_1 = module_0.rex(var_0)
    var_2 = 'start_middle_end'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'not_start'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = 'startend'
    var_7 = var_1(var_6)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test(\\d)?'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'test1'
    var_5 = var_1(var_4)
    assert var_5 is True
    var_6 = 'test123'
    var_7 = var_1(var_6)
    assert var_7 is True
    var_8 = 'tes'
    var_9 = var_1(var_8)
    assert var_9 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 7/11 statements.
# Partially parsed test_do_to_path_single_level_path_with_key. Retrieved 8/13 statements.
# Partially parsed test_do_to_path_nested_path. Retrieved 8/17 statements.
# Partially parsed test_do_to_path_with_unary_predicate. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 11/14 statements.
# Partially parsed test_do_to_path_discard_command. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_with_missing_key_creates_empty_pmap. Retrieved 6/13 statements.
# Partially parsed test_do_to_path_multiple_keys_matching_predicate. Retrieved 12/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0]
    var_6 = 2
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1]
    var_6 = 99
    var_7 = {var_2: var_6}

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
    var_11 = lambda x: x * var_10

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0]
    var_5 = 99
    var_6 = lambda x: var_5
    var_7 = module_0._do_to_path(var_3, var_4, var_6)
    var_8 = var_7[1]
    assert var_8 == 99

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 'a'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 1
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = [var_8]
    var_10 = 5
    var_11 = lambda x: x + var_10



# Parsed testcases at query #7
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
    var_5 = lambda i: i > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20), (2, 30)])
    assert var_7 is True

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
    var_4 = lambda i, v: v >= var_1
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_predicate_callable_check.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_evaluates_to_false. Retrieved 12/36 statements.


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
    var_13 = module_0._items(var_7)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'



# Parsed testcases at query #13
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
    var_7 = lambda idx: idx in var_6
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
    var_4 = lambda idx, val: val >= var_1
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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._get_keys_and_values(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 10



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
    var_7 = module_0._items(var_6)
    var_8 = [(k, v) for (k, v) in var_7 if key_spec(k)]
    var_9 = bool(var_8 == [])
    assert var_9 is True



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #17
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 1), (1, 2), (2, 3)])
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 7/55 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 5/53 statements.
# Failed to parse test_do_to_path_with_path_and_callable_command.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = lambda x: x
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 42
    var_5 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #21
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
    var_1 = 15
    var_2 = 10
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val > var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 15), (3, 20)])
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 7/33 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'name'
    var_7 = module_0._get_keys_and_values(var_5, var_6)
    var_8 = bool(var_7 == [('name', 'John')])
    assert var_8 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_predicate_callable_command.


def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    var_2 = var_0 if var_1 else var_0
    assert var_2 == 42
    var_3 = callable(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/41 statements.


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
    var_11 = bool(var_10 == [('a', 1), ('b', 2)])
    assert var_11 is True
    var_12 = len(var_10)
    var_13 = bool(var_12 > 0)
    assert var_13 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 1/7 statements.
# Partially parsed test_get_keys_and_values_with_object_missing_attribute. Retrieved 1/6 statements.


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
    var_5 = 2
    var_6 = 0
    var_7 = lambda i: i % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = 35
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 10
    var_6 = lambda i, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(1, 15), (2, 25), (3, 35)])
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

def test_case_0():
    var_0 = 'attr1'

def test_case_0():
    var_0 = 'missing_attr'



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/46 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #29
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = bool(var_4 == [(0, 1), (1, 2), (2, 3)])
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 28/55 statements.


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
    var_20 = 'x'
    var_21 = 'y'
    var_22 = 'z'
    var_23 = 10
    var_24 = 20
    var_25 = 30
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = 15
    var_28 = lambda k, v: v > var_27
    var_29 = module_0._get_keys_and_values(var_26, var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 'y'
    var_32 = 20
    var_33 = (var_31, var_32)
    var_34 = bool(('y', 20) in var_29)
    assert var_34 is True
    var_35 = 'z'
    var_36 = 30
    var_37 = (var_35, var_36)
    var_38 = bool(('z', 30) in var_29)
    assert var_38 is True
    var_39 = 40
    var_40 = [var_23, var_24, var_25, var_39]
    var_41 = 0
    var_42 = lambda idx: idx % var_5 == var_41
    var_43 = module_0._get_keys_and_values(var_40, var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = 0
    var_46 = 10
    var_47 = (var_45, var_46)
    var_48 = bool((0, 10) in var_43)
    assert var_48 is True
    var_49 = 2
    var_50 = 30
    var_51 = (var_49, var_50)
    var_52 = bool((2, 30) in var_43)
    assert var_52 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/26 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 8/56 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 5/53 statements.
# Failed to parse test_do_to_path_with_nested_path.


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



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_predicate_at_line_1_evaluates_to_false.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_default.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_keyword_only.




# Parsed testcases at query #35
#--------------------------

# Failed to parse test_items_with_custom_dict_like_object.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'a'
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = bool(('a', 1) in var_6)
    assert var_11 is True
    var_12 = 'b'
    var_13 = 2
    var_14 = (var_12, var_13)
    var_15 = bool(('b', 2) in var_6)
    assert var_15 is True

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



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = 'not_callable_string'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 1/7 statements.
# Partially parsed test_get_keys_and_values_with_object_missing_attribute. Retrieved 1/6 statements.


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
    var_5 = lambda idx, val: val >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(2, 15), (3, 20)])
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

def test_case_0():
    var_0 = 'attr1'

def test_case_0():
    var_0 = 'missing_attr'



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
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'a'
    var_12 = 1
    var_13 = (var_11, var_12)
    var_14 = bool(('a', 1) in var_9)
    assert var_14 is True
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = bool(('c', 3) in var_9)
    assert var_18 is True
    var_19 = callable(var_8)
    var_20 = bool(var_19)
    assert var_20 is True
    var_21 = var_8(var_0)
    assert var_21 is True
    var_22 = var_8(var_1)
    assert var_22 is False



# Parsed testcases at query #39
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
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'a'
    var_12 = 1
    var_13 = (var_11, var_12)
    var_14 = bool(('a', 1) in var_9)
    assert var_14 is True
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = bool(('c', 3) in var_9)
    assert var_18 is True
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 'z'
    var_22 = 10
    var_23 = 20
    var_24 = 5
    var_25 = {var_19: var_22, var_20: var_23, var_21: var_24}
    var_26 = 8
    var_27 = lambda k, v: v > var_26
    var_28 = module_0._get_keys_and_values(var_25, var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'x'
    var_31 = 10
    var_32 = (var_30, var_31)
    var_33 = bool(('x', 10) in var_28)
    assert var_33 is True
    var_34 = 'y'
    var_35 = 20
    var_36 = (var_34, var_35)
    var_37 = bool(('y', 20) in var_28)
    assert var_37 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/36 statements.


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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_evaluates_to_false. Retrieved 12/33 statements.


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



# Parsed testcases at query #42
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_callable_command. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_discard. Retrieved 5/11 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_nested_path. Retrieved 7/16 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 12/16 statements.
# Partially parsed test_update_structure_preserves_unchanged_values. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_empty_sentinel_creates_pmap. Retrieved 5/13 statements.


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
    var_11 = 'a'
    var_12 = 'b'

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
    var_9 = lambda x: x * var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 99

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 5

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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = [var_2]
    var_4 = 42
    var_5 = 'b'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'simple_string'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 14/19 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = 'some'
    var_8 = 'path'
    var_9 = [var_7, var_8]
    var_10 = None
    var_11 = lambda e, k: var_10
    var_12 = lambda e, k: e.discard(k) if k in e else var_10
    var_13 = var_11 is var_12



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_false. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = 'some'
    var_8 = 'path'
    var_9 = [var_7, var_8]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_update_structure_empty_path_with_discard. Retrieved 12/17 statements.
# Partially parsed test_update_structure_empty_path_with_callable. Retrieved 12/19 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 8/17 statements.
# Partially parsed test_update_structure_single_kvs_no_path. Retrieved 7/11 statements.
# Partially parsed test_update_structure_multiple_kvs_no_path_with_discard. Retrieved 11/16 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_command. Retrieved 8/15 statements.
# Partially parsed test_update_structure_preserves_structure_when_no_change. Retrieved 10/15 statements.


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
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = (var_1,)
    var_6 = [var_5]
    var_7 = 5

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 42

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = (var_1, var_2)
    var_8 = [var_6, var_7]
    var_9 = []
    var_10 = [var_1, var_3]

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
    var_6 = 5
    var_7 = {var_0: var_1, var_3: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 1
    var_9 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = 'some'
    var_9 = 'path'
    var_10 = [var_8, var_9]



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 13/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = (var_0, var_5)
    var_7 = (var_1, var_5)
    var_8 = [var_6, var_7]
    var_9 = []
    var_10 = 'x'
    var_11 = [var_10]
    var_12 = []



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_items_with_dict.
# Failed to parse test_items_with_empty_dict.


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
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {('key1', 'value1'), ('key2', 'value2')})
    assert var_7 is True



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_one_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_defaults.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #55
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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda idx, val: val >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {(1, 20), (2, 30)})
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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_callable_key_spec_with_arity_1. Retrieved 12/39 statements.


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



# Parsed testcases at query #57
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #58
#--------------------------

# Partially parsed test_do_to_path_empty_path_with_callable_command. Retrieved 5/51 statements.
# Partially parsed test_do_to_path_empty_path_with_non_callable_command. Retrieved 4/50 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = lambda x: x + var_1
    var_3 = 5
    var_4 = []
    var_5 = module_0._do_to_path(var_3, var_4, var_2)
    assert var_5 == 6

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = []
    var_3 = 42
    var_4 = module_0._do_to_path(var_1, var_2, var_3)
    assert var_4 == 42



# Parsed testcases at query #59
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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 25
    var_6 = lambda idx, val: val >= var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(2, 30), (3, 40)])
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



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 1/7 statements.
# Partially parsed test_get_keys_and_values_with_object_missing_attribute. Retrieved 1/7 statements.


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
    var_5 = lambda i, v: v >= var_2
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

def test_case_0():
    var_0 = 'attr1'

def test_case_0():
    var_0 = 'missing_attr'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_empty_path_and_command. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 12/20 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 12/15 statements.
# Partially parsed test_update_structure_discard_with_sentinel_value. Retrieved 7/13 statements.
# Partially parsed test_update_structure_with_empty_pmap_expansion. Retrieved 8/14 statements.
# Partially parsed test_update_structure_reversed_order_for_discard. Retrieved 11/15 statements.


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
    var_11 = 'a'
    var_12 = 'b'

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
    var_10 = 5
    var_11 = lambda v: v + var_10

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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = []
    var_7 = 'x'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = 10
    var_7 = lambda v: var_6

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = (var_1, var_2)
    var_8 = (var_3, var_4)
    var_9 = [var_6, var_7, var_8]
    var_10 = []



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = None
    var_7 = ''



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_items_with_dict.
# Failed to parse test_items_with_empty_dict.
# Failed to parse test_items_with_dict_multiple_items.


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



# Parsed testcases at query #65
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
    var_7 = bool(var_6 == [(2, 30), (3, 40)])
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



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'b'
    var_2 = 'args'
    var_3 = 'kwargs'



# Parsed testcases at query #67
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
    var_5 = lambda idx, val: val >= var_1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 10), (2, 15), (3, 20)])
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



# Parsed testcases at query #68
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
    var_4 = 0
    var_5 = lambda i: i > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = set(var_6)
    var_8 = bool(var_7 == {(1, 20), (2, 30)})
    assert var_8 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda i, v: v >= var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {(1, 20), (2, 30)})
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



# Parsed testcases at query #69
#--------------------------




def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/61 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 10/66 statements.
# Failed to parse test_update_structure_with_empty_sentinel_and_command.


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
    var_8 = []
    var_9 = lambda x: x
    var_10 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_value_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 12/23 statements.
# Partially parsed test_update_structure_discard_with_empty_sentinel. Retrieved 8/14 statements.
# Partially parsed test_update_structure_creates_new_pmap_for_empty_sentinel. Retrieved 9/20 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 14/18 statements.
# Partially parsed test_update_structure_with_vector. Retrieved 11/15 statements.
# Partially parsed test_update_structure_discard_vector_reverse_order. Retrieved 13/17 statements.


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
    var_8 = lambda x: x * var_3
    var_9 = {var_0: var_3, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 99
    var_9 = {var_0: var_8, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 2
    var_8 = {var_2: var_4, var_3: var_5}
    var_9 = [var_2]
    var_10 = 100
    var_11 = {var_2: var_10, var_3: var_5}

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
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = 'y'
    var_6 = [var_5]
    var_7 = 5
    var_8 = {var_1: var_2, var_5: var_7}

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
    var_9 = (var_2, var_5)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = 99
    var_13 = {var_0: var_12, var_1: var_12, var_2: var_12}

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
    var_10 = [var_9, var_9, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = (var_3, var_4)
    var_7 = (var_1, var_2)
    var_8 = 0
    var_9 = (var_8, var_0)
    var_10 = [var_6, var_7, var_9]
    var_11 = []
    var_12 = [var_1, var_3]



# Parsed testcases at query #72
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #73
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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/37 statements.


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



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_get_arity.




# Parsed testcases at query #76
#--------------------------

# Failed to parse test_predicate_at_line_1_evaluates_to_false.




# Parsed testcases at query #77
#--------------------------




def test_case_0():
    var_0 = 'some_string_key'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = 'some'
    var_9 = 'path'
    var_10 = [var_8, var_9]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_callable_command. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_non_empty_path. Retrieved 11/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_command. Retrieved 7/12 statements.
# Partially parsed test_update_structure_preserves_unchanged_keys. Retrieved 11/14 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 12/15 statements.
# Partially parsed test_update_structure_discard_multiple_keys_reversed. Retrieved 13/17 statements.


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
    var_8 = lambda x: x * var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 99

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
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = 42

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
    var_10 = 10

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
    var_11 = 50

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
    var_9 = (var_2, var_5)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = {}



# Parsed testcases at query #80
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
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'a'
    var_12 = 1
    var_13 = (var_11, var_12)
    var_14 = bool(('a', 1) in var_9)
    assert var_14 is True
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = bool(('c', 3) in var_9)
    assert var_18 is True

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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = 10
    var_12 = (var_10, var_11)
    var_13 = bool((0, 10) in var_8)
    assert var_13 is True
    var_14 = 2
    var_15 = 30
    var_16 = (var_14, var_15)
    var_17 = bool((2, 30) in var_8)
    assert var_17 is True

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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'b'
    var_11 = 2
    var_12 = (var_10, var_11)
    var_13 = bool(('b', 2) in var_8)
    assert var_13 is True
    var_14 = 'c'
    var_15 = 3
    var_16 = (var_14, var_15)
    var_17 = bool(('c', 3) in var_8)
    assert var_17 is True

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
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = 1
    var_10 = 20
    var_11 = (var_9, var_10)
    var_12 = bool((1, 20) in var_7)
    assert var_12 is True
    var_13 = 2
    var_14 = 30
    var_15 = (var_13, var_14)
    var_16 = bool((2, 30) in var_7)
    assert var_16 is True
    var_17 = 3
    var_18 = 40
    var_19 = (var_17, var_18)
    var_20 = bool((3, 40) in var_7)
    assert var_20 is True

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



# Parsed testcases at query #81
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



# Parsed testcases at query #82
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_all_default_parameters.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_keyword_only_parameters.




# Parsed testcases at query #83
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'
    var_2 = 'kwargs'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 1/7 statements.


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
    var_5 = 15
    var_6 = lambda i, v: v > var_5
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
    var_9 = 'must take 1 or 2 arguments'

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

def test_case_0():
    var_0 = 'x'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard. Retrieved 9/15 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable. Retrieved 12/19 statements.
# Partially parsed test_update_structure_with_empty_path_and_value. Retrieved 9/13 statements.
# Partially parsed test_update_structure_preserves_structure_when_no_changes. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 8/19 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 14/19 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 12/18 statements.


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
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}

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
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = lambda x: x
    var_11 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = [var_0]
    var_5 = 20
    var_6 = 20
    var_7 = {var_0: var_6}

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
    var_11 = 99
    var_12 = 99
    var_13 = {var_0: var_12, var_1: var_12, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_2, var_5)
    var_8 = (var_0, var_3)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = {var_1: var_4}



# Parsed testcases at query #87
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



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/41 statements.


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
    var_31 = bool(True)
    assert var_31 is True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = ()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_default.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_keyword_only.




# Parsed testcases at query #3
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
    var_0 = '^\\d{3}-\\d{4}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123-4567'
    var_3 = var_1(var_2)
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}-\\d{4}$'
    var_1 = module_0.rex(var_0)
    var_2 = '12-456'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^Test'
    var_1 = module_0.rex(var_0)
    var_2 = 'Test_string'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'test_string'
    var_5 = var_1(var_4)
    assert var_5 is False

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
    var_0 = '^[a-z]+@[a-z]+\\.[a-z]+'
    var_1 = module_0.rex(var_0)
    var_2 = 'test@example.com'
    var_3 = var_1(var_2)
    assert var_3 is True



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
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda i, v: v >= var_1
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 10), (2, 15), (3, 20)])
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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 100
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_items_with_dict. Retrieved 7/9 statements.
# Failed to parse test_items_with_custom_items_method.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = dict(var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
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
    var_2 = dict(var_1)
    var_3 = bool(var_2 == {})
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 42
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/22 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 8/36 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = module_0._get_keys_and_values(var_5, var_6)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True
    var_9 = callable(var_6)
    assert var_9 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 2/5 statements.
# Partially parsed test_get_keys_and_values_with_object_missing_attribute. Retrieved 1/5 statements.


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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'y')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 5
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda i: i > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 'y'), (2, 'z')])
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = lambda i, v: v in var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 'y'), (2, 'z')])
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

def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'missing'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #12
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
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'a'
    var_12 = 1
    var_13 = (var_11, var_12)
    var_14 = bool(('a', 1) in var_9)
    assert var_14 is True
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = bool(('c', 3) in var_9)
    assert var_18 is True

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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = 10
    var_12 = (var_10, var_11)
    var_13 = bool((0, 10) in var_8)
    assert var_13 is True
    var_14 = 2
    var_15 = 30
    var_16 = (var_14, var_15)
    var_17 = bool((2, 30) in var_8)
    assert var_17 is True

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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'b'
    var_11 = 2
    var_12 = (var_10, var_11)
    var_13 = bool(('b', 2) in var_8)
    assert var_13 is True
    var_14 = 'c'
    var_15 = 3
    var_16 = (var_14, var_15)
    var_17 = bool(('c', 3) in var_8)
    assert var_17 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 2
    var_9 = 30
    var_10 = (var_8, var_9)
    var_11 = bool((2, 30) in var_6)
    assert var_11 is True
    var_12 = 3
    var_13 = 40
    var_14 = (var_12, var_13)
    var_15 = bool((3, 40) in var_6)
    assert var_15 is True

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
    var_2 = lambda idx, val: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_do_to_path_single_level_with_dict_key. Retrieved 8/18 statements.
# Partially parsed test_do_to_path_multi_level_path. Retrieved 9/23 statements.
# Partially parsed test_do_to_path_with_unary_predicate. Retrieved 14/18 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 14/18 statements.
# Partially parsed test_do_to_path_with_list_structure. Retrieved 9/13 statements.
# Partially parsed test_do_to_path_with_discard_command. Retrieved 7/11 statements.
# Partially parsed test_do_to_path_nested_with_predicate. Retrieved 17/27 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = lambda x: var_5
    var_7 = []
    var_8 = module_0._do_to_path(var_2, var_7, var_6)
    var_9 = bool(var_8 == {'b': 2})
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = module_0._do_to_path(var_2, var_6, var_5)
    var_8 = bool(var_7 == {'b': 2})
    assert var_8 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_0]
    var_7 = {var_1: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 99
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1]
    var_8 = {var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = lambda x: x * var_7
    var_9 = [var_0, var_1]
    var_10 = lambda k: k in var_9
    var_11 = [var_10]
    var_12 = 20
    var_13 = {var_0: var_7, var_1: var_12, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = lambda x: x * var_7
    var_9 = lambda k, v: v > var_3
    var_10 = [var_9]
    var_11 = 20
    var_12 = 30
    var_13 = {var_0: var_3, var_1: var_11, var_2: var_12}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = lambda x: x * var_4
    var_6 = [var_0]
    var_7 = 20
    var_8 = [var_0, var_7, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = {var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = 100
    var_11 = lambda x: x + var_10
    var_12 = lambda k: k == var_2
    var_13 = [var_0, var_12]
    var_14 = 101
    var_15 = {var_2: var_14, var_3: var_5}
    var_16 = {var_2: var_7, var_3: var_8}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_true. Retrieved 14/37 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = callable(var_2)
    assert var_3 is True
    var_4 = lambda x, y: x > var_1
    var_5 = callable(var_4)
    assert var_5 is True
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = {var_6: var_9, var_7: var_10, var_8: var_11}
    var_13 = module_0._get_keys_and_values(var_12, var_2)
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = bool(var_13 == [(1, 'a'), (2, 'b'), (3, 'c')])
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_items_with_dict.


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
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = dict(var_5)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_get_arity.




# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_defaults.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_kwargs.




# Parsed testcases at query #22
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
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._items(var_2)
    var_4 = dict(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = bool(var_2 == [(0, 42)])
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/38 statements.


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
    var_20 = bool(var_10 == [('a', 1), ('c', 3)])
    assert var_20 is True



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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'y')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = module_0._get_keys_and_values(var_2, var_3)
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda idx: idx > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 'y'), (2, 'z')])
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = lambda idx, val: val in var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [(1, 'y'), (2, 'z')])
    assert var_8 is True

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
    var_5 = 10
    var_6 = lambda k, v: v > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [])
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_attribute. Retrieved 1/6 statements.
# Partially parsed test_get_keys_and_values_with_object_missing_attribute. Retrieved 1/5 statements.


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
    var_8 = var_6[0][1].__class__.__name__
    assert var_8 == '_EMPTY_SENTINEL'

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
    var_6 = var_4[0][1].__class__.__name__
    assert var_6 == '_EMPTY_SENTINEL'

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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 15
    var_6 = lambda i, v: v > var_5
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
    var_1 = 'key'
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = var_2[0][0]
    assert var_3 == 'key'
    var_4 = var_2[0][1].__class__.__name__
    assert var_4 == '_EMPTY_SENTINEL'

def test_case_0():
    var_0 = 'attr'

def test_case_0():
    var_0 = 'missing'



# Parsed testcases at query #27
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
    var_3 = lambda x, y, z: x
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'callable in transform path must take 1 or 2 arguments'

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



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_get_arity.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/34 statements.


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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_empty_path_and_value_command. Retrieved 11/16 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_non_discard_command. Retrieved 7/15 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 9/18 statements.
# Partially parsed test_update_structure_discard_with_sentinel_value. Retrieved 8/15 statements.
# Partially parsed test_update_structure_multiple_kvs_reverse_order. Retrieved 11/16 statements.


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
    var_10 = 11
    var_11 = {var_0: var_10, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 100
    var_9 = 100
    var_10 = {var_0: var_9, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = []
    var_6 = 5
    var_7 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = 10

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = (var_3, var_4)
    var_6 = (var_1, var_2)
    var_7 = 0
    var_8 = (var_7, var_0)
    var_9 = [var_5, var_6, var_8]
    var_10 = []



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 12/35 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 10
    var_5 = 20
    var_6 = 30
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = [var_1, var_3]
    var_9 = lambda k: k in var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = callable(var_9)
    assert var_11 is True
    var_12 = len(var_10)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 10
    var_15 = (var_13, var_14)
    var_16 = bool(('a', 10) in var_10)
    assert var_16 is True
    var_17 = 'c'
    var_18 = 30
    var_19 = (var_17, var_18)
    var_20 = bool(('c', 30) in var_10)
    assert var_20 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = bool(not var_7)
    assert var_8 is True



# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_get_keys_and_values_with_object_and_attribute. Retrieved 1/7 statements.
# Partially parsed test_get_keys_and_values_with_object_and_missing_attribute. Retrieved 1/6 statements.


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
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 5

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
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = set(var_8)
    var_10 = bool(var_9 == {(0, 10), (2, 30)})
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = 20
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda idx, val: val >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = set(var_6)
    var_8 = bool(var_7 == {(2, 15), (3, 20)})
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

def test_case_0():
    var_0 = 'attr1'

def test_case_0():
    var_0 = 'missing'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_is_callable. Retrieved 12/39 statements.


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
    var_13 = len(var_10)
    assert var_13 == 2
    var_14 = 'a'
    var_15 = 1
    var_16 = (var_14, var_15)
    var_17 = bool(('a', 1) in var_10)
    assert var_17 is True
    var_18 = 'c'
    var_19 = 3
    var_20 = (var_18, var_19)
    var_21 = bool(('c', 3) in var_10)
    assert var_21 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_value_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 12/23 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_non_discard_command. Retrieved 7/13 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 13/17 statements.
# Partially parsed test_update_structure_discard_multiple_kvs_reversed. Retrieved 13/17 statements.


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
    var_10 = 11
    var_11 = {var_0: var_10, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 99
    var_9 = {var_0: var_8, var_1: var_3}

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
    var_11 = {var_2: var_10, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 5
    var_6 = {var_0: var_1, var_3: var_5}

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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_2, var_5)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = 0
    var_12 = {var_0: var_11, var_1: var_4, var_2: var_11}

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
    var_9 = (var_2, var_5)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = {}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_value_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 12/23 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard. Retrieved 6/11 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 8/14 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 13/17 statements.
# Partially parsed test_update_structure_reversed_order_for_discard. Retrieved 11/15 statements.


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
    var_8 = lambda x: x * var_3
    var_9 = {var_0: var_3, var_1: var_3}

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
    var_9 = {var_0: var_8, var_1: var_3}

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
    var_11 = {var_2: var_10, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 5

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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_2, var_5)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = 10
    var_12 = {var_0: var_11, var_1: var_4, var_2: var_11}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 0
    var_6 = (var_5, var_0)
    var_7 = (var_1, var_2)
    var_8 = (var_3, var_4)
    var_9 = [var_6, var_7, var_8]
    var_10 = []



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = (var_0, var_5)
    var_7 = (var_1, var_5)
    var_8 = [var_6, var_7]
    var_9 = []



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_keys_and_values_with_dict_and_string_key. Retrieved 8/29 statements.
# Partially parsed test_get_keys_and_values_with_dict_and_unary_predicate. Retrieved 10/31 statements.
# Partially parsed test_get_keys_and_values_with_dict_and_binary_predicate. Retrieved 9/30 statements.
# Partially parsed test_get_keys_and_values_with_list_and_index. Retrieved 6/27 statements.
# Failed to parse test_get_keys_and_values_with_invalid_key.


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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_false. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = [var_4, var_5]
    var_7 = []
    var_8 = None
    var_9 = lambda e, k: var_8



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 13/22 statements.


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
    var_12 = (var_10, var_11)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 13/24 statements.


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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/44 statements.


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



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 'not_callable'
    var_1 = callable(var_0)
    assert var_1 is False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 20/43 statements.


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
    var_12 = lambda k, v: v > var_4
    var_13 = module_0._get_keys_and_values(var_7, var_12)
    var_14 = bool(var_13 == [('b', 2), ('c', 3)])
    assert var_14 is True
    var_15 = 10
    var_16 = 20
    var_17 = 30
    var_18 = 40
    var_19 = [var_15, var_16, var_17, var_18]
    var_20 = 0
    var_21 = lambda idx: idx % var_5 == var_20
    var_22 = module_0._get_keys_and_values(var_19, var_21)
    var_23 = bool(var_22 == [(0, 10), (2, 30)])
    assert var_23 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 8/29 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_key'
    var_2 = 'test_key'
    var_3 = 'other_key'
    var_4 = 'test_value'
    var_5 = 'other_value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = callable(var_1)
    assert var_8 is False
    var_9 = bool(var_7 == [('test_key', 'test_value')])
    assert var_9 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/26 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #49
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



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_update_structure_predicate_line_4. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]
    var_9 = bool(not var_5)
    assert var_9 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_evaluates_to_false. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'some'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = []
    var_7 = None
    var_8 = lambda x, y: var_7
    var_9 = 'x'
    var_10 = [var_9]
    var_11 = lambda x, y: var_7



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_get_keys_and_values_predicate_evaluates_to_false. Retrieved 11/32 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda k: var_0
    var_2 = '_get_arity'
    var_3 = None
    var_4 = '_items'
    var_5 = '_get'
    var_6 = {}
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = '_get_arity'
    var_10 = '_items'
    var_11 = '_get'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 11/38 statements.


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
    var_20 = bool(var_10[0][0] == 'a' or var_10[0][0] == 'c')
    assert var_20 is True



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #55
#--------------------------

# Partially parsed test_callable_predicate_evaluates_to_false. Retrieved 13/36 statements.


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
    var_8 = False
    var_9 = lambda k: var_8
    var_10 = module_0._get_keys_and_values(var_7, var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_13 = lambda k, v: var_8
    var_14 = module_0._get_keys_and_values(var_12, var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_get_keys_and_values_callable_predicate_evaluates_to_true. Retrieved 14/41 statements.


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
    var_12 = lambda k, v: v > var_4
    var_13 = module_0._get_keys_and_values(var_7, var_12)
    var_14 = bool(var_13 == [('b', 2), ('c', 3)])
    assert var_14 is True
    var_15 = callable(var_9)
    assert var_15 is True
    var_16 = callable(var_12)
    assert var_16 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_callable_command. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 11/22 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 16/20 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/14 statements.
# Partially parsed test_update_structure_discard_multiple_in_reverse. Retrieved 13/17 statements.
# Partially parsed test_update_structure_with_nested_empty_sentinel. Retrieved 9/17 statements.


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
    var_8 = lambda x: x * var_3
    var_9 = {var_0: var_3, var_1: var_3}

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
    var_9 = 10
    var_10 = {var_0: var_9, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_3}
    var_7 = [var_2]
    var_8 = 5
    var_9 = 5
    var_10 = {var_2: var_9}

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
    var_11 = 10
    var_12 = lambda x: x + var_11
    var_13 = 11
    var_14 = 13
    var_15 = {var_0: var_13, var_1: var_4, var_2: var_14}

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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_3)
    var_8 = (var_1, var_4)
    var_9 = (var_2, var_5)
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = {}

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



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_empty_path_and_value_command. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 13/25 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_vector. Retrieved 13/17 statements.
# Partially parsed test_update_structure_discard_multiple_elements_in_reverse. Retrieved 13/17 statements.
# Partially parsed test_update_structure_preserves_unchanged_values. Retrieved 11/14 statements.


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
    var_7 = lambda x: x * var_3
    var_8 = []
    var_9 = {var_0: var_3, var_1: var_3}

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
    var_9 = {var_0: var_8, var_1: var_3}

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
    var_9 = 100
    var_10 = lambda x: x + var_9
    var_11 = []
    var_12 = {var_2: var_4, var_3: var_5}

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
    var_6 = 'b'

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
    var_10 = lambda x: x * var_9
    var_11 = 20
    var_12 = [var_9, var_11, var_2]

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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'args'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key. Retrieved 14/35 statements.


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
    var_12 = 10
    var_13 = 20
    var_14 = 30
    var_15 = [var_12, var_13, var_14]
    var_16 = 0
    var_17 = lambda i: i > var_16
    var_18 = module_0._get_keys_and_values(var_15, var_17)
    var_19 = sorted(var_18)
    var_20 = bool(var_19 == [(1, 20), (2, 30)])
    assert var_20 is True

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
    var_11 = 10
    var_12 = 20
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = lambda i, v: v >= var_12
    var_16 = module_0._get_keys_and_values(var_14, var_15)
    var_17 = sorted(var_16)
    var_18 = bool(var_17 == [(1, 20), (2, 30)])
    assert var_18 is True

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



# Parsed testcases at query #61
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
    var_11 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_12 = lambda k, v: v > var_3
    var_13 = module_0._get_keys_and_values(var_11, var_12)
    var_14 = bool(var_13 == [('b', 2), ('c', 3)])
    assert var_14 is True
    var_15 = 10
    var_16 = 20
    var_17 = 30
    var_18 = 40
    var_19 = [var_15, var_16, var_17, var_18]
    var_20 = 0
    var_21 = lambda idx: idx % var_4 == var_20
    var_22 = module_0._get_keys_and_values(var_19, var_21)
    var_23 = bool(var_22 == [(0, 10), (2, 30)])
    assert var_23 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_true. Retrieved 9/19 statements.


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



# Parsed testcases at query #63
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
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0][0]
    assert var_8 == 'x'

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
    var_5 = 2
    var_6 = 0
    var_7 = lambda idx: idx % var_5 == var_6
    var_8 = module_0._get_keys_and_values(var_4, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == [(0, 10), (2, 30)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = 35
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 10
    var_6 = lambda idx, val: val > var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = sorted(var_7)
    var_9 = bool(var_8 == [(1, 15), (2, 25), (3, 35)])
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
    var_7 = 'callable in transform path must take 1 or 2 arguments'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 10/36 statements.


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
    var_10 = bool(var_9 == [('a', 1)])
    assert var_10 is True
    var_11 = callable(var_8)
    assert var_11 is False



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_update_structure_predicate_line_4_false. Retrieved 17/20 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = 'some'
    var_8 = 'path'
    var_9 = [var_7, var_8]
    var_10 = 'discard'
    var_11 = bool(not (not var_9 and var_10 is 'discard'))
    assert var_11 is True
    var_12 = (var_0, var_2)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'some_other_command'
    var_16 = bool(not (not var_14 and var_15 is 'some_other_command'))
    assert var_16 is True
    var_17 = [var_7, var_8]
    var_18 = 'other_command'
    var_19 = bool(not (not var_17 and var_18 is 'other_command'))
    assert var_19 is True



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_get_arity_no_parameters.
# Failed to parse test_get_arity_single_required_parameter.
# Failed to parse test_get_arity_multiple_required_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_mixed_required_and_optional.
# Failed to parse test_get_arity_with_var_args.
# Failed to parse test_get_arity_with_keyword_only.
# Failed to parse test_get_arity_with_kwargs.




# Parsed testcases at query #67
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
    var_10 = bool(var_9 == [('a', 1), ('c', 3)])
    assert var_10 is True

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
    var_5 = lambda k, v: v >= var_2
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [(2, 30), (3, 40)])
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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 100
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [])
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



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 7/31 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = module_0._get_keys_and_values(var_5, var_6)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_callable_key_spec_with_arity_1. Retrieved 11/38 statements.


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
    var_20 = var_10[0][0]
    assert var_20 == 'a'
    var_21 = var_10[1][0]
    assert var_21 == 'c'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_true. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = []



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_empty_path_and_callable_command. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_callable_command. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_nested_path_and_discard. Retrieved 11/22 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_callable_command. Retrieved 6/15 statements.
# Partially parsed test_update_structure_with_multiple_kvs. Retrieved 15/19 statements.
# Partially parsed test_update_structure_with_nested_path_and_value_change. Retrieved 9/22 statements.
# Partially parsed test_update_structure_preserves_other_keys. Retrieved 11/14 statements.
# Partially parsed test_update_structure_with_vector_and_discard. Retrieved 10/14 statements.


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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = 100

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
    var_10 = {var_3: var_5}

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
    var_11 = 10
    var_12 = lambda x: x * var_11
    var_13 = 20
    var_14 = {var_0: var_11, var_1: var_13, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = []
    var_6 = 5
    var_7 = {var_1: var_6}
    var_8 = {var_1: var_6}

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = (var_0, var_1)
    var_6 = (var_2, var_3)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = [var_0, var_2]



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 8/35 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = module_0._get_keys_and_values(var_5, var_6)
    var_8 = bool(var_7 == [('a', 1)])
    assert var_8 is True
    var_9 = callable(var_6)
    assert var_9 is False



