####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_mixed_types.
# Failed to parse test_get_arity_all_required_positional.
# Failed to parse test_get_arity_only_keyword_args.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_get_arity_no_params.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_params.
# Failed to parse test_get_arity_varargs_and_varkw.
# Failed to parse test_get_arity_keyword_only.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_invalid_arity_zero. Retrieved 3/7 statements.
# Partially parsed test_get_keys_and_values_with_invalid_arity_three. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'banana')])
    assert var_6 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_object_attribute_success. Retrieved 2/7 statements.
# Partially parsed test_get_object_attribute_error_returns_default. Retrieved 2/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
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
    var_6 = 99
    var_7 = module_0._get(var_4, var_5, var_6)
    assert var_7 == 99

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 0
    var_6 = module_0._get(var_3, var_4, var_5)
    assert var_6 == 20

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = -1
    var_6 = module_0._get(var_3, var_4, var_5)
    assert var_6 == -1

def test_case_0():
    var_0 = 'x'
    var_1 = 0

def test_case_0():
    var_0 = 'y'
    var_1 = 'fallback'

def test_case_0():
    pass

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 'error'
    var_3 = module_0._get(var_0, var_1, var_2)
    assert var_3 == 'h'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hi'
    var_1 = 5
    var_2 = 'missing'
    var_3 = module_0._get(var_0, var_1, var_2)
    assert var_3 == 'missing'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_do_to_path_identity. Retrieved 4/7 statements.
# Partially parsed test_do_to_path_replace_value. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_nested_replace. Retrieved 6/11 statements.
# Partially parsed test_do_to_path_with_callable_command. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_with_predicate_key. Retrieved 14/20 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 11/19 statements.
# Partially parsed test_do_to_path_with_sequence. Retrieved 7/15 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = []
    var_3 = lambda x: x

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 10
    var_5 = 10

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 2
    var_5 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = lambda x: x + var_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = lambda k: k in var_5
    var_7 = 'val'
    var_8 = [var_6, var_7]
    var_9 = 99
    var_10 = lambda k: k == var_3
    var_11 = 'inner'
    var_12 = [var_10, var_11]
    var_13 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda k, v: v > var_0
    var_3 = 'val'
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = 'x'
    var_7 = lambda k, v: v[var_6] > var_0
    var_8 = [var_7, var_6]
    var_9 = 99
    var_10 = 99

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 0
    var_3 = 'a'
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = 10

def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = lambda x, y, z: var_1
    var_3 = [var_2]
    var_4 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_update_structure_with_discard_on_pmap. Retrieved 10/14 statements.
# Partially parsed test_update_structure_with_discard_on_pvector. Retrieved 11/15 statements.
# Partially parsed test_update_structure_with_replacement_on_pmap. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_expansion_on_pmap. Retrieved 8/14 statements.
# Partially parsed test_update_structure_nested_path_replacement. Retrieved 9/19 statements.
# Partially parsed test_update_structure_with_no_op_command. Retrieved 8/11 statements.


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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = (var_4, var_0)
    var_6 = 1
    var_7 = (var_6, var_1)
    var_8 = [var_5, var_7]
    var_9 = []
    var_10 = [var_2]

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 10
    var_6 = lambda x: var_5
    var_7 = {var_0: var_1, var_3: var_5}

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x
    var_7 = {var_0: var_1}



# Parsed testcases at query #7
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

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
    var_3 = 'z'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'z'



# Parsed testcases at query #8
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #9
#--------------------------




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
    var_8 = module_0._items(var_4)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = (var_2,)
    var_4 = 0
    var_5 = (var_0, var_1)
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = module_0._items(var_3)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_no_params.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_params.
# Failed to parse test_get_arity_varargs_and_varkw.
# Failed to parse test_get_arity_keyword_only_ignored.




# Parsed testcases at query #11
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #12
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'z'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 5
    var_4 = 2
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: len(k) > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('banana', 2), ('cherry', 10)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('b', 10), ('c', 5)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)
    var_8 = bool(var_7 == [(0, 'zero'), (2, 'two')])
    assert var_8 is True

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
    var_4 = lambda a, b, c: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_items_predicate_evaluates_to_false.




# Parsed testcases at query #14
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'abc'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = ''
    var_7 = var_1(var_6)
    assert var_7 is False
    var_8 = 123
    var_9 = var_1(var_8)
    assert var_9 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-zA-Z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'Hello'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'hello123'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = None
    var_7 = var_1(var_6)
    assert var_7 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^start'
    var_1 = module_0.rex(var_0)
    var_2 = 'start_of_string'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'the_start'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[!@#]$'
    var_1 = module_0.rex(var_0)
    var_2 = '!'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = '@'
    var_5 = var_1(var_4)
    assert var_5 is True
    var_6 = 'a'
    var_7 = var_1(var_6)
    assert var_7 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_do_to_path_identity. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_replace_value. Retrieved 15/21 statements.
# Partially parsed test_do_to_path_nested_replace. Retrieved 9/16 statements.
# Partially parsed test_do_to_path_with_predicate. Retrieved 18/23 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 14/17 statements.
# Partially parsed test_do_to_path_discard_logic. Retrieved 10/18 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 8/14 statements.
# Partially parsed test_do_to_path_with_vector. Retrieved 7/10 statements.
# Partially parsed test_do_to_path_with_non_existent_key. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 'arg'
    var_9 = 11
    var_10 = {var_0: var_9, var_8: var_3}
    var_11 = {var_0: var_2}
    var_12 = [var_0]
    var_13 = lambda x: x + var_6
    var_14 = {var_0: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 10
    var_6 = lambda x: x + var_5
    var_7 = 11
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = [var_7]
    var_9 = 10
    var_10 = lambda x: x + var_9
    var_11 = 'name_b'
    var_12 = 12
    var_13 = {var_0: var_3, var_11: var_12, var_2: var_5}
    var_14 = lambda k: k == var_1
    var_15 = [var_14]
    var_16 = lambda x: x + var_9
    var_17 = {var_0: var_3, var_1: var_12, var_2: var_5}

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
    var_11 = 20
    var_12 = 30
    var_13 = {var_0: var_3, var_1: var_11, var_2: var_12}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = (var_0, var_2)
    var_7 = [var_6]
    var_8 = []
    var_9 = {var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = lambda x: x
    var_7 = {var_0: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0]
    var_4 = 10
    var_5 = lambda x: x + var_4
    var_6 = 12

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'nonexistent'
    var_4 = [var_3]
    var_5 = lambda x: x
    var_6 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_items_predicate_is_false. Retrieved 6/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._items(var_2)
    var_4 = (var_0, var_1)
    var_5 = [var_4]



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_params.
# Failed to parse test_get_arity_all_defaults.
# Failed to parse test_get_arity_keyword_only_no_default.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #20
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 'non_existent'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #22
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_items_evaluates_to_False_at_line_4. Retrieved 9/10 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0._items(var_5)
    var_7 = (var_4, var_0)
    var_8 = [var_7]



# Parsed testcases at query #24
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_key_spec. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_do_to_path_identity. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_direct_value. Retrieved 5/7 statements.
# Partially parsed test_do_to_path_single_level_update. Retrieved 9/12 statements.
# Partially parsed test_do_to_path_nested_update. Retrieved 8/15 statements.
# Partially parsed test_do_to_path_with_callable_command. Retrieved 8/11 statements.
# Partially parsed test_do_to_path_with_predicate_path. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_binary_predicate_path. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_discard_command. Retrieved 8/18 statements.
# Partially parsed test_do_to_path_with_list_index_access. Retrieved 7/10 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = 10
    var_7 = 10
    var_8 = {var_0: var_7, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 10
    var_6 = 10
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = 5
    var_5 = lambda x: x + var_4
    var_6 = 6
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = [var_7]
    var_9 = 99
    var_10 = 99
    var_11 = {var_0: var_3, var_1: var_10, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v == var_4
    var_8 = [var_7]
    var_9 = 99
    var_10 = 99
    var_11 = {var_0: var_3, var_1: var_10, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = [var_0]
    var_7 = {var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 0
    var_4 = [var_3]
    var_5 = 10
    var_6 = 10

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = [var_4]
    var_6 = 10



# Parsed testcases at query #28
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = lambda x: x
    var_5 = module_0._do_to_path(var_2, var_3, var_4)



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
    var_7 = lambda k: k == var_0 or k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1), ('b', 2)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'mapping'
    var_2 = 'c'
    var_3 = 1
    var_4 = 10
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('mapping', 10)])
    assert var_10 is True

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
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)
    var_6 = bool(var_5 == [('name', 'test')])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'nonexistent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'nonexistent'

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
    var_0 = 'id'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = 'name'
    var_6 = 'foo'
    var_7 = (var_5, var_6)
    var_8 = [var_2, var_4, var_7]
    var_9 = lambda k, v: v == var_3
    var_10 = module_0._get_keys_and_values(var_8, var_9)
    var_11 = bool(var_10 == [(2, 30)])
    assert var_11 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k, v: k == var_4 and v == var_1
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 20)])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(var_4 == [(0, 10)])
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_get_arity_skips_parameters_with_defaults.




# Parsed testcases at query #31
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #32
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 5
    var_4 = 2
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = lambda k: k.startswith(var_7)
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('apple', 5)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 5
    var_4 = 2
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('apple', 5), ('cherry', 10)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'one')])
    assert var_7 is True

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
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'b'



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #34
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

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
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'non_existent'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_update_structure_discard_mapping. Retrieved 8/13 statements.
# Partially parsed test_update_structure_discard_mapping_missing_key. Retrieved 9/14 statements.
# Partially parsed test_update_structure_update_value. Retrieved 7/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 6/16 statements.
# Partially parsed test_update_structure_with_empty_sentinel_expansion. Retrieved 7/12 statements.


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
    var_1 = 'b'
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = 'a'
    var_7 = (var_6, var_0)
    var_8 = [var_7]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = 2
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 3
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = 2
    var_6 = lambda x: var_5



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path. Retrieved 7/19 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some'
    var_2 = 'path'
    var_3 = [var_1, var_2]
    var_4 = 'discard'
    var_5 = 'mock_globals'
    var_6 = None
    var_7 = []
    var_8 = {}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_discard_pmap. Retrieved 7/15 statements.
# Partially parsed test_update_structure_discard_vector. Retrieved 9/12 statements.
# Partially parsed test_update_structure_set_value. Retrieved 6/14 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 5/13 statements.
# Partially parsed test_update_structure_no_change. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'c'
    var_4 = (var_3, var_1)
    var_5 = 'b'
    var_6 = [var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = (var_0, var_1)
    var_6 = (var_1, var_2)
    var_7 = [var_4, var_5, var_6]
    var_8 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 2
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = 10
    var_4 = lambda x: var_3

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = lambda x: x



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_predicate_false_due_to_path. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = (var_4, var_5)
    var_7 = lambda x, y: x



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path. Retrieved 4/35 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_key'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = []
    var_5 = {}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_do_to_path_direct_value. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 5



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_do_to_path_no_path_returns_command_result. Retrieved 9/14 statements.
# Partially parsed test_do_to_path_with_path_and_value_update. Retrieved 9/17 statements.
# Partially parsed test_do_to_path_with_path_and_callable_command. Retrieved 9/17 statements.
# Partially parsed test_do_to_path_with_predicate_in_path. Retrieved 22/32 statements.
# Partially parsed test_do_to_path_with_discard_command. Retrieved 9/24 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = lambda x: x
    var_8 = {var_3: var_4}
    var_9 = []
    var_10 = 5

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_4]
    var_8 = 10
    var_9 = 10
    var_10 = {var_4: var_9}

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_4]
    var_8 = lambda x: x + var_5
    var_9 = 2
    var_10 = {var_4: var_9}

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = lambda k: k == var_4
    var_11 = 'inner'
    var_12 = [var_10, var_11]
    var_13 = {var_3: var_6, var_4: var_7}
    var_14 = lambda k: k == var_3
    var_15 = 'target'
    var_16 = [var_14, var_15]
    var_17 = 5
    var_18 = {var_4: var_17}
    var_19 = lambda k: k == var_3
    var_20 = [var_19, var_4]
    var_21 = 10
    var_22 = 10
    var_23 = {var_4: var_22}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = [var_0, var_1]
    var_6 = 10
    var_7 = 10
    var_8 = {var_1: var_7}



# Parsed testcases at query #2
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = lambda x: x
    var_5 = module_0._do_to_path(var_2, var_3, var_4)



# Parsed testcases at query #3
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = lambda x: x
    var_5 = module_0._do_to_path(var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_mixed_args.
# Failed to parse test_get_arity_with_keyword_only.
# Failed to parse test_get_arity_with_var_args.




# Parsed testcases at query #5
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
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(var_4 == [(0, 'first')])
    assert var_5 is True

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
    var_3 = lambda x: x
    var_4 = None
    var_5 = lambda : var_4
    var_6 = module_0._get_keys_and_values(var_2, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'non_existent'



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
    var_7 = lambda k: k == var_0 or k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('a', 1), ('b', 2)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 10
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('b', 10)])
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = lambda k: k == var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)
    var_6 = bool(var_5 == [(0, 'apple')])
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
    var_4 = lambda a, b, c: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 'z'
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_mixed_params.
# Failed to parse test_get_arity_ignores_keyword_only.
# Failed to parse test_get_arity_ignores_varargs_and_varkw.
# Failed to parse test_get_arity_complex_mix.




# Parsed testcases at query #8
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_key_spec. Retrieved 8/13 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = True
    var_6 = lambda k: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)
    var_8 = bool(var_7 == [('a', 1), ('b', 2)])
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = lambda x: x
    var_5 = module_0._do_to_path(var_2, var_3, var_4)



# Parsed testcases at query #11
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_0)
    assert var_2 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'def'
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'abc'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False
    var_4 = None
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = [var_0]
    var_7 = var_1(var_6)
    assert var_7 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'zabc'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'anything'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'abc'
    var_5 = module_0.rex(var_4)
    var_6 = var_5(var_0)
    assert var_6 is False



# Parsed testcases at query #12
#--------------------------




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
    var_8 = module_0._items(var_4)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hi'
    var_1 = 0
    var_2 = 'h'
    var_3 = (var_1, var_2)
    var_4 = 1
    var_5 = 'i'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._items(var_0)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_key_spec. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #14
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 'apple'), (1, 'banana')])
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)
    var_4 = bool(var_3 == [(0, 10), (1, 20)])
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
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_mixed_args.
# Failed to parse test_get_arity_with_keyword_only.
# Failed to parse test_get_arity_var_args_and_kwargs.
# Failed to parse test_get_arity_complex_case.




# Parsed testcases at query #17
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #18
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

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
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'non_existent'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_predicate. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'other_key'
    var_2 = 'target_value'
    var_3 = 'other_value'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_key_spec. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 2
    var_10 = 'b'
    var_11 = (var_9, var_10)
    var_12 = 4
    var_13 = 'd'
    var_14 = (var_12, var_13)
    var_15 = 1
    var_16 = 'a'
    var_17 = (var_15, var_16)



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_callable_key'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = {var_0: var_1}
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = module_0._get_keys_and_values(var_5, var_7)
    var_9 = bool(var_4 != var_8)
    assert var_9 is True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = lambda k: len(k) > var_4
    var_6 = 1
    var_7 = lambda k: k == var_6
    var_8 = module_0._get_keys_and_values(var_3, var_7)
    var_9 = bool(var_8 == [(1, 'banana')])
    assert var_9 is True

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
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 'non_existent'



# Parsed testcases at query #24
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(var_4 == [(0, 'apple')])
    assert var_5 is True

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
    var_4 = 10
    var_5 = 2
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = bool(var_9 == [('b', 10)])
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda i, v: v == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 20)])
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_args.
# Failed to parse test_get_arity_keyword_only_ignored.
# Failed to parse test_get_arity_varargs_ignored.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_structure_with_discard_on_pmap. Retrieved 10/13 statements.
# Partially parsed test_update_structure_with_replacement_on_pmap. Retrieved 11/14 statements.
# Partially parsed test_update_structure_with_nested_path_on_pmap. Retrieved 9/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel_expansion. Retrieved 11/16 statements.
# Partially parsed test_update_structure_with_discard_and_missing_key. Retrieved 9/12 statements.
# Partially parsed test_update_structure_with_path_and_command_returning_same_value. Retrieved 13/25 statements.


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
    var_9 = {var_1: var_3}

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
    var_9 = lambda x: var_8
    var_10 = {var_0: var_8, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'inner'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 99
    var_7 = lambda x: var_6
    var_8 = {var_1: var_6}

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'b'
    var_7 = (var_6, var_2)
    var_8 = [var_7]
    var_9 = []
    var_10 = 5
    var_11 = lambda x: var_10
    var_12 = {var_3: var_4, var_6: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x
    var_7 = 'inner'
    var_8 = {var_7: var_1}
    var_9 = {var_7: var_1}
    var_10 = [var_7]
    var_11 = lambda x: x
    var_12 = {var_7: var_1}



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #28
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #29
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

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
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'non_existent'



# Parsed testcases at query #30
#--------------------------




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
    var_8 = module_0._items(var_4)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hi'
    var_1 = 0
    var_2 = 'h'
    var_3 = (var_1, var_2)
    var_4 = 1
    var_5 = 'i'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._items(var_0)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_items_predicate_is_false. Retrieved 6/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._items(var_2)
    var_4 = (var_0, var_1)
    var_5 = [var_4]



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #33
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_update_structure_predicate_is_false_when_path_exists. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = (var_4, var_5)
    var_7 = lambda x, y: x
    var_8 = None
    var_9 = lambda x, y: var_8
    var_10 = {var_0: var_1}



# Parsed testcases at query #35
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0][0]
    assert var_6 == 'b'

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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 8/12 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = (var_4, var_5)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path. Retrieved 8/11 statements.
# Partially parsed test_update_structure_predicate_false_via_command. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = (var_4, var_5)
    var_7 = lambda x: x

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = ()
    var_5 = lambda x: x



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_predicate_false_by_path_exists. Retrieved 14/27 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = (var_4, var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_0.object(*var_7, **var_8)
    var_10 = 'module'
    var_11 = []
    var_12 = {}
    var_13 = []
    var_14 = {}
    var_15 = 'a'
    var_16 = 'b'
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = (var_5,)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'discard_op'
    var_1 = []
    var_2 = ''
    var_3 = 'discard_op'
    var_4 = bool(not var_2 is False)
    assert var_4 is True
    var_5 = bool(var_3 is var_0)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------




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
    var_8 = module_0._items(var_4)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = (var_3, var_0)
    var_5 = 1
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0._items(var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hi'
    var_1 = 0
    var_2 = 'h'
    var_3 = (var_1, var_2)
    var_4 = 1
    var_5 = 'i'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._items(var_0)
    var_9 = list(var_8)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #44
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



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.




# Parsed testcases at query #46
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_structure_discard_pmap. Retrieved 10/14 statements.
# Partially parsed test_update_structure_discard_vector_index. Retrieved 11/15 statements.
# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 7/20 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 9/13 statements.


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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = (var_4, var_0)
    var_6 = 2
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = []
    var_10 = [var_1]

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
    var_0 = 'b'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = [var_0]
    var_5 = 2
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 10
    var_7 = lambda x: var_6
    var_8 = {var_0: var_6}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_update_structure_discard_pmap. Retrieved 11/16 statements.
# Partially parsed test_update_structure_discard_pvector. Retrieved 14/19 statements.
# Partially parsed test_update_structure_set_value. Retrieved 9/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel. Retrieved 6/15 statements.
# Partially parsed test_update_structure_no_change. Retrieved 7/9 statements.
# Partially parsed test_update_structure_error_on_invalid_arity. Retrieved 6/12 statements.


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
    var_9 = {var_0: var_2, var_1: var_3}
    var_10 = {}

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
    var_12 = [var_0, var_1, var_2]
    var_13 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 2
    var_7 = lambda v: var_6
    var_8 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = 10
    var_4 = lambda v: var_3
    var_5 = {var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda v: v

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = lambda v: v



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #50
#--------------------------

# Partially parsed test_items_predicate_is_false_with_dict. Retrieved 11/14 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)
    var_6 = 'items'
    var_7 = hasattr(var_5, var_6)
    var_8 = (var_0, var_2)
    var_9 = (var_1, var_3)
    var_10 = [var_8, var_9]



# Parsed testcases at query #51
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [(1, 'b')])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'name'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = bool(var_6 == [('name', 'Alice')])
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = bool(var_6 == [(1, 'banana')])
    assert var_7 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




