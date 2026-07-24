####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_do_to_path_identity_with_no_path. Retrieved 5/7 statements.
# Partially parsed test_do_to_path_with_value_and_no_path. Retrieved 7/11 statements.
# Partially parsed test_do_to_path_nested_update_pmap. Retrieved 7/14 statements.
# Partially parsed test_do_to_path_with_callable_command. Retrieved 7/16 statements.
# Partially parsed test_do_to_path_with_predicate_key. Retrieved 12/15 statements.
# Partially parsed test_do_to_path_with_binary_predicate_key. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 2
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 2
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 2
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = (var_0, var_1)
    var_8 = lambda k: k in var_7
    var_9 = [var_8]
    var_10 = 10
    var_11 = {var_0: var_10, var_1: var_10, var_2: var_5}

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
    var_10 = {var_0: var_3, var_1: var_9, var_2: var_9}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_keys_and_values_with_unary_callable. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_binary_callable. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_invalid_arity_callable. Retrieved 3/7 statements.


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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
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
    var_0 = 'first'
    var_1 = 'second'
    var_2 = 'third'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'second')])
    assert var_6 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #3
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = 'abcd'
    var_5 = var_1(var_4)
    assert var_5 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'def'
    var_3 = var_1(var_2)
    assert var_3 is False
    var_4 = 'ab'
    var_5 = var_1(var_4)
    assert var_5 is False

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
    var_6 = 'abc'
    var_7 = [var_6]
    var_8 = var_1(var_7)
    assert var_8 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}-\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123-456'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = '12-345'
    var_5 = var_1(var_4)
    assert var_5 is False
    var_6 = 'abc-def'
    var_7 = var_1(var_6)
    assert var_7 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = ' '
    var_5 = var_1(var_4)
    assert var_5 is False



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_params.
# Failed to parse test_get_arity_keyword_only_ignored.
# Failed to parse test_get_arity_varargs_ignored.




# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_do_to_path_evaluates_true_when_path_is_empty. Retrieved 4/1 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []



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



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_do_to_path_direct_value. Retrieved 5/7 statements.
# Partially parsed test_do_to_path_direct_command. Retrieved 6/8 statements.
# Partially parsed test_do_to_path_nested_update. Retrieved 7/14 statements.
# Partially parsed test_do_to_path_with_discard_command. Retrieved 9/16 statements.
# Partially parsed test_do_to_path_with_predicate_key. Retrieved 11/14 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 2
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 2
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = [var_0, var_1]
    var_7 = 10
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
    var_10 = {var_1: var_9}

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_items_evaluates_to_false_on_dict. Retrieved 6/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 9/10 statements.


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
    var_3 = 1
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(var_4 == [(1, 'second')])
    assert var_5 is True

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = lambda k: k.startswith(var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = bool(var_8 == [('banana', 2), ('cherry', 3)])
    assert var_9 is True

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = bool(var_5 == [(1, 'b')])
    assert var_6 is True

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

# Partially parsed test_do_to_path_evaluates_true_when_path_is_empty. Retrieved 5/2 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = []

def test_case_0():
    var_0 = 'success'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = []



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




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_do_to_path_evaluates_true_when_path_is_empty. Retrieved 4/1 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_keys_and_values_with_dict_key. Retrieved 6/7 statements.
# Partially parsed test_get_keys_and_values_with_list_index. Retrieved 5/6 statements.
# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 8/9 statements.
# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 9/10 statements.
# Partially parsed test_get_keys_and_values_with_invalid_arity_zero. Retrieved 5/7 statements.
# Partially parsed test_get_keys_and_values_with_invalid_arity_three. Retrieved 5/7 statements.
# Partially parsed test_get_keys_and_values_with_non_existent_key_returns_sentinel. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda a, b, c: var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_items_returns_items_method_when_available. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



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
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = bool(var_4 == [(1, 'one')])
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
    var_3 = True
    var_4 = lambda k, v, x: var_3
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



# Parsed testcases at query #24
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
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_update_structure_discard_key_in_pmap. Retrieved 6/17 statements.
# Partially parsed test_update_structure_insert_value_in_pmap. Retrieved 5/13 statements.
# Partially parsed test_update_structure_expand_empty_path_with_value. Retrieved 5/13 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_discard_non_existent_key. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 10
    var_3 = 10
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 5
    var_3 = 5
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 2
    var_7 = 2
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'non_existent'
    var_2 = []
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = {var_3: var_4}
    var_8 = [var_3]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_structure_predicate_false_by_path_exists. Retrieved 12/15 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = None
    var_4 = lambda e, k: var_3
    var_5 = lambda v, path, command: v
    var_6 = 'key'
    var_7 = 'value'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = 'some'
    var_11 = 'path'
    var_12 = (var_10, var_11)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path. Retrieved 9/20 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = (var_4,)
    var_6 = lambda v, p, c: v
    var_7 = []
    var_8 = {}
    var_9 = module_0.object(*var_7, **var_8)
    var_10 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_update_structure_discard_mapping. Retrieved 10/14 statements.
# Partially parsed test_update_structure_discard_mapping_non_existent. Retrieved 9/13 statements.
# Partially parsed test_update_structure_update_value. Retrieved 9/13 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/20 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 9/17 statements.
# Partially parsed test_update_structure_no_change_if_result_is_same. Retrieved 8/12 statements.


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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
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
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = {var_0: var_6}

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
    var_0 = 'empty'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = []
    var_6 = 10
    var_7 = lambda x: var_6
    var_8 = {var_1: var_2, var_4: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1
    var_7 = {var_0: var_1}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = ''
    var_1 = []



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_types.
# Failed to parse test_get_arity_keyword_only_ignored.
# Failed to parse test_get_arity_varargs_ignored.




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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 'b'

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #3
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_do_to_path_base_case_value. Retrieved 5/10 statements.
# Partially parsed test_do_to_path_base_case_direct_value. Retrieved 6/9 statements.
# Partially parsed test_do_to_path_with_path_and_update. Retrieved 12/25 statements.
# Partially parsed test_do_to_path_with_predicate_key. Retrieved 11/15 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 7/15 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = []

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 10

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = 'key_not_exists'
    var_8 = [var_3, var_7]
    var_9 = 1
    var_10 = {var_4: var_9}
    var_11 = [var_3]
    var_12 = 2
    var_13 = {var_4: var_12}

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = lambda k: k == var_3
    var_9 = [var_8]
    var_10 = 10
    var_11 = 10
    var_12 = {var_3: var_11}

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 1
    var_7 = 'ValueError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.




# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_keys_and_values_with_unary_predicate. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 7/10 statements.
# Partially parsed test_get_keys_and_values_with_list_structure_and_unary_predicate. Retrieved 4/7 statements.


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
    var_4 = 10
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = var_4[0][0]
    assert var_5 == 'non_existent'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_get_arity_no_params.
# Failed to parse test_get_arity_single_param.
# Failed to parse test_get_arity_multiple_params.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_keyword_only.
# Failed to parse test_get_arity_mixed_types.
# Failed to parse test_get_arity_positional_only.




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
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)
    var_6 = bool(var_5 == [('name', 'Alice')])
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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_get_arity_skips_parameters_with_defaults.




# Parsed testcases at query #12
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
    var_0 = '^\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = var_1(var_2)
    assert var_3 is True
    var_4 = '1234'
    var_5 = var_1(var_4)
    assert var_5 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    assert var_3 is False

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = None
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
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = var_1(var_2)
    assert var_3 is True



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'executed'
    var_3 = lambda x: var_2
    var_4 = module_0._do_to_path(var_0, var_1, var_3)
    var_5 = var_4 == var_2

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'static_value'
    var_3 = module_0._do_to_path(var_0, var_1, var_2)
    var_4 = var_3 == var_2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_items_predicate_evaluates_to_false. Retrieved 4/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._items(var_2)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_6 = 'items'
    var_7 = hasattr(var_3, var_6)
    var_8 = bool(not var_7)
    assert var_8 is True



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



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = bool(var_3 == [('a', 1)])
    assert var_4 is True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



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

# Partially parsed test_update_structure_discard_mapping. Retrieved 10/13 statements.
# Partially parsed test_update_structure_replace_mapping. Retrieved 11/14 statements.
# Partially parsed test_update_structure_nested_update_mapping. Retrieved 9/19 statements.
# Partially parsed test_update_structure_vector_discard_order. Retrieved 13/16 statements.
# Partially parsed test_update_structure_no_change_if_same_value. Retrieved 8/11 statements.


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
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = {var_1: var_6}

def test_case_0():
    pass

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
    var_12 = [var_0, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1
    var_7 = {var_0: var_1}



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_structure_discard_map. Retrieved 8/11 statements.
# Partially parsed test_update_structure_update_map_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update_map. Retrieved 7/15 statements.
# Partially parsed test_update_structure_with_empty_sentinel_expansion. Retrieved 6/12 statements.
# Partially parsed test_update_structure_discard_non_existent_key. Retrieved 3/8 statements.
# Partially parsed test_update_structure_vector_index_discard. Retrieved 13/16 statements.


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
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = [var_3]
    var_5 = []
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 10
    var_5 = lambda x: x + var_4
    var_6 = 11

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_2]
    var_4 = 100
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_discard_key. Retrieved 9/13 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 6/14 statements.
# Partially parsed test_update_structure_no_change_if_value_same. Retrieved 8/11 statements.
# Partially parsed test_update_structure_discard_non_existent_key_in_path. Retrieved 8/19 statements.


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
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 10
    var_7 = lambda x: var_6
    var_8 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = 5
    var_4 = lambda x: var_3
    var_5 = {var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1
    var_7 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = {var_1: var_2}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_structure_predicate_false_due_to_path. Retrieved 13/34 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = None
    var_4 = lambda e, k: var_3
    var_5 = 'key'
    var_6 = 'value'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = 'some'
    var_10 = 'path'
    var_11 = [var_9, var_10]
    var_12 = '__main__'
    var_13 = {var_5: var_6}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_update_structure_discard_mapping. Retrieved 10/14 statements.
# Partially parsed test_update_structure_discard_mapping_partial. Retrieved 9/13 statements.
# Partially parsed test_update_structure_update_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 10/20 statements.
# Partially parsed test_update_structure_no_change_if_value_same. Retrieved 8/11 statements.
# Partially parsed test_update_structure_vector_discard_reverse_order. Retrieved 13/17 statements.


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
    var_1 = 'key_to_remove'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_1, var_3)
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_0: var_2}

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
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = {var_1: var_8}

def test_case_0():
    pass

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x
    var_7 = {var_0: var_1}

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
    var_12 = [var_2]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_discard_key. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_non_existent_key. Retrieved 6/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_expand_empty_sentinel. Retrieved 6/14 statements.
# Partially parsed test_update_structure_vector_discard_reverse_order. Retrieved 10/14 statements.
# Partially parsed test_update_structure_no_change_if_value_same. Retrieved 8/11 statements.


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
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_1: var_3}

def test_case_0():
    var_0 = 'b'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = []
    var_5 = {var_0: var_1}

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
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = 10
    var_4 = lambda x: var_3
    var_5 = {var_1: var_3}

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
    var_9 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1
    var_7 = {var_0: var_1}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_structure_predicate_false_via_path_exists. Retrieved 7/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_1,)
    var_3 = None
    var_4 = []
    var_5 = (var_1,)
    var_6 = None



# Parsed testcases at query #34
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



