####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/9 statements.


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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 10
    var_8 = [var_0, var_2]
    var_9 = module_0._do_to_path(var_6, var_8, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 10
    var_8 = 'd'
    var_9 = [var_0, var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = lambda k: k == var_0
    var_9 = [var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = lambda k, v: v == var_4
    var_9 = [var_8]
    var_10 = module_0._do_to_path(var_6, var_9, var_7)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0]



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
    var_7 = lambda k: k == var_0
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
    var_7 = lambda k, v: v > var_3
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
    var_7 = module_0._get_keys_and_values(var_6, var_0)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_rex_lambda_matches_string. Retrieved 2/3 statements.
# Partially parsed test_rex_lambda_no_match. Retrieved 3/4 statements.
# Partially parsed test_rex_lambda_non_string_input. Retrieved 3/4 statements.
# Partially parsed test_rex_with_complex_pattern. Retrieved 5/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.rex(var_0)
    var_2 = 'other'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.rex(var_0)
    var_2 = 123

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'ABC'
    var_4 = 'a1'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #6
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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #8
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_with_mixed_args.




# Parsed testcases at query #11
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/13 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 16/22 statements.
# Partially parsed test__update_structure_with_empty_path_and_non_discard_command. Retrieved 12/15 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 19/22 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_discard_command. Retrieved 8/14 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_non_discard_command. Retrieved 8/15 statements.
# Partially parsed test__update_structure_with_partial_path_and_command. Retrieved 26/29 statements.


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
    var_15 = {var_5: var_6}

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
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]
    var_15 = lambda x: x * var_6
    var_16 = {var_2: var_6}
    var_17 = {var_5: var_6}
    var_18 = {var_0: var_16, var_1: var_17}

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
    var_7 = lambda x: x * var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = {var_3: var_4}
    var_12 = {var_2: var_11}
    var_13 = (var_0, var_12)
    var_14 = {var_3: var_7}
    var_15 = {var_2: var_14}
    var_16 = (var_1, var_15)
    var_17 = [var_13, var_16]
    var_18 = [var_2, var_3]
    var_19 = lambda x: x * var_7
    var_20 = {var_3: var_7}
    var_21 = {var_2: var_20}
    var_22 = 4
    var_23 = {var_3: var_22}
    var_24 = {var_2: var_23}
    var_25 = {var_0: var_21, var_1: var_24}



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



# Parsed testcases at query #17
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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Failed to parse test__get_arity_with_no_positional_parameters.




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



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #22
#--------------------------

# Failed to parse test__get_arity_with_all_defaults.




# Parsed testcases at query #23
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)



# Parsed testcases at query #24
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_4. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = []
    var_7 = []



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
    var_7 = lambda k: k == var_0
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
    var_7 = lambda k, v: v > var_3
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
    var_7 = module_0._get_keys_and_values(var_6, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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



# Parsed testcases at query #27
#--------------------------

# Failed to parse test__get_arity_with_all_default_parameters.




# Parsed testcases at query #28
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
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
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 9/14 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_1, var_3)
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_1, var_3)
    var_6 = []
    var_7 = lambda x: x * var_3
    var_8 = 4

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
    var_9 = lambda x: x
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'path'
    var_6 = [var_5]
    var_7 = 'some_command'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/12 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 14/17 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_non_discard_command. Retrieved 8/12 statements.
# Partially parsed test__update_structure_with_empty_sentinal_and_discard_command. Retrieved 6/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 11/16 statements.


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
    var_11 = lambda x: x * var_5
    var_12 = module_0._update_structure(var_6, var_9, var_10, var_11)
    var_13 = {var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 2
    var_6 = lambda x: var_5
    var_7 = {var_0: var_1, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = {var_0: var_1}

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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_predicate_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = 'some_command'



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []
    var_3 = None



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_positional_arg.
# Failed to parse test_get_arity_with_multiple_positional_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_mixed_args.
# Failed to parse test_get_arity_with_varargs.
# Failed to parse test_get_arity_with_kwargs.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #39
#--------------------------

# Failed to parse test__get_arity_with_no_args.
# Failed to parse test__get_arity_with_one_arg.
# Failed to parse test__get_arity_with_multiple_args.
# Failed to parse test__get_arity_with_default_args.
# Failed to parse test__get_arity_with_keyword_only_args.
# Failed to parse test__get_arity_with_positional_only_args.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_structure. Retrieved 1/7 statements.


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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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

def test_case_0():
    var_0 = 'x'



# Parsed testcases at query #41
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = var_4 == var_5



# Parsed testcases at query #42
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



# Parsed testcases at query #43
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test__update_structure_empty_path_and_discard_command. Retrieved 12/14 statements.
# Partially parsed test__update_structure_non_empty_path_and_discard_command. Retrieved 17/19 statements.
# Partially parsed test__update_structure_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test__update_structure_empty_sentinel_and_non_discard_command. Retrieved 6/11 statements.
# Partially parsed test__update_structure_with_pmap_leaf_node. Retrieved 7/13 statements.


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
    var_11 = {var_2: var_4, var_3: var_5}
    var_12 = (var_0, var_11)
    var_13 = {var_2: var_7, var_3: var_8}
    var_14 = (var_1, var_13)
    var_15 = [var_12, var_14]
    var_16 = [var_2]

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = {var_2: var_3}
    var_9 = (var_0, var_8)
    var_10 = {var_2: var_5}
    var_11 = (var_1, var_10)
    var_12 = [var_9, var_11]
    var_13 = [var_2]
    var_14 = 3
    var_15 = lambda x: x * var_14
    var_16 = module_0._update_structure(var_7, var_12, var_13, var_15)

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
    var_4 = 'c'
    var_5 = [var_4]
    var_6 = lambda x: x



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/13 statements.


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



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_one_arg.
# Failed to parse test_get_arity_multiple_args.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_keyword_only.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_mixed.




# Parsed testcases at query #47
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #48
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



# Parsed testcases at query #49
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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
    var_7 = {var_0, var_1}
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 17/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 8/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_empty_sentinel. Retrieved 16/19 statements.


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
    var_8 = 4
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = {var_2: var_4, var_3: var_5}
    var_12 = (var_0, var_11)
    var_13 = {var_2: var_7, var_3: var_8}
    var_14 = (var_1, var_13)
    var_15 = [var_12, var_14]
    var_16 = [var_2]

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
    var_11 = {var_2: var_4, var_3: var_5}
    var_12 = (var_0, var_11)
    var_13 = {var_2: var_7, var_3: var_8}
    var_14 = (var_1, var_13)
    var_15 = [var_12, var_14]
    var_16 = [var_2]
    var_17 = lambda x: x * var_5
    var_18 = module_0._update_structure(var_10, var_15, var_16, var_17)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = []
    var_8 = lambda x: x + var_2

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
    var_11 = {var_2: var_4, var_3: var_5}
    var_12 = (var_0, var_11)
    var_13 = 'c'
    var_14 = [var_2]
    var_15 = lambda x: x * var_5



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = []
    var_7 = []
    var_8 = 'discard'



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_get_arity_with_all_default_parameters.




# Parsed testcases at query #55
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



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #58
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)



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
    var_7 = False
    var_8 = lambda k: var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 12/14 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 5/9 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/10 statements.
# Partially parsed test__update_structure_with_pmap_leaf_node. Retrieved 7/12 statements.


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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = {var_2: var_3}
    var_9 = (var_0, var_8)
    var_10 = {var_2: var_5}
    var_11 = (var_1, var_10)
    var_12 = [var_9, var_11]
    var_13 = [var_2]
    var_14 = 3
    var_15 = lambda x: x * var_14
    var_16 = module_0._update_structure(var_7, var_12, var_13, var_15)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = lambda x: x



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #63
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_two_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #64
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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
    var_7 = module_0._get_keys_and_values(var_6, var_0)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_1
    var_5 = module_0._get_keys_and_values(var_3, var_4)



# Parsed testcases at query #67
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



# Parsed testcases at query #68
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'some'
    var_6 = 'path'
    var_7 = (var_5, var_6)
    var_8 = 'some_command'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = {}
    var_6 = []
    var_7 = []
    var_8 = 'discard'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
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
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
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
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.
# Failed to parse test_get_arity_with_mixed_args.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = lambda x: x.clear()
    var_7 = module_0._do_to_path(var_4, var_5, var_6)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = [var_0]
    var_8 = 'd'
    var_9 = 4
    var_10 = {var_8: var_9}
    var_11 = lambda x: x.update(var_10)
    var_12 = module_0._do_to_path(var_6, var_7, var_11)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'd'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = [var_0, var_2]
    var_10 = 'e'
    var_11 = 4
    var_12 = {var_10: var_11}
    var_13 = lambda x: x.update(var_12)
    var_14 = module_0._do_to_path(var_8, var_9, var_13)

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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k: k == var_1
    var_8 = [var_7]
    var_9 = lambda x: x * var_4
    var_10 = module_0._do_to_path(var_6, var_8, var_9)

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
    var_8 = [var_7]
    var_9 = lambda x: x * var_5
    var_10 = module_0._do_to_path(var_6, var_8, var_9)



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0._items(var_0)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test__get_arity_returns_false_for_optional_positional.




# Parsed testcases at query #11
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
    var_7 = (var_0, var_1)
    var_8 = lambda k: k in var_7
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
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_items_with_non_dict_structure. Retrieved 5/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #13
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_get_arity_with_all_default_parameters.




# Parsed testcases at query #15
#--------------------------

# Failed to parse test__get_arity_with_all_params_having_defaults.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 13/15 statements.
# Partially parsed test__update_structure_with_empty_value_and_non_discard_command. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1: var_2}
    var_6 = (var_0, var_5)
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = lambda x: x + var_2
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)

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



# Parsed testcases at query #17
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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



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
    var_7 = lambda k: k.startswith(var_0)
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
    var_7 = lambda k, v: v > var_3
    var_8 = module_0._get_keys_and_values(var_6, var_7)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = callable(var_0)



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

# Partially parsed test_predicate_evaluates_to_true. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'MockStructure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'MockEvolver'
    var_4 = ()
    var_5 = 'persistent'
    var_6 = None
    var_7 = lambda : var_6
    var_8 = {var_5: var_7}
    var_9 = []
    var_10 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 13/15 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 7/11 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/12 statements.
# Partially parsed test_update_structure_with_pmap_as_leaf_node. Retrieved 9/15 statements.


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
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = lambda x: x



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []
    var_3 = 'some_command'



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'path'
    var_6 = [var_5]
    var_7 = 'some_command'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 7/10 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.object()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 10/14 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 18/22 statements.
# Partially parsed test__update_structure_with_empty_path_and_non_discard_command. Retrieved 12/15 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 19/22 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 8/14 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/15 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_empty_sentinel. Retrieved 14/21 statements.
# Partially parsed test__update_structure_with_no_changes. Retrieved 11/14 statements.


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
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]
    var_15 = {}
    var_16 = {var_5: var_6}
    var_17 = {var_0: var_15, var_1: var_16}

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
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = {var_2: var_3}
    var_10 = (var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = (var_1, var_11)
    var_13 = [var_10, var_12]
    var_14 = [var_2]
    var_15 = lambda x: x * var_6
    var_16 = {var_2: var_6}
    var_17 = {var_5: var_6}
    var_18 = {var_0: var_16, var_1: var_17}

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
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 'c'
    var_10 = [var_2]
    var_11 = lambda x: x * var_6
    var_12 = {var_2: var_3}
    var_13 = {var_5: var_6}

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
    var_10 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/13 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 9/19 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 5/11 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 6/12 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 9/20 statements.


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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = lambda x: x + var_2
    var_7 = 2
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 1
    var_3 = 0
    var_4 = {var_0: var_3}

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
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = {var_2: var_4}



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_get_arity_with_default_parameters.




# Parsed testcases at query #32
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



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
    var_7 = [var_0, var_1]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 11/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 5/9 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/10 statements.
# Partially parsed test_update_structure_with_pmap_leaf_node. Retrieved 3/10 statements.


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
    var_9 = lambda x: x * var_3
    var_10 = module_0._update_structure(var_4, var_7, var_8, var_9)

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
    var_11 = lambda x: x * var_5
    var_12 = module_0._update_structure(var_6, var_9, var_10, var_11)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = lambda x: x



# Parsed testcases at query #35
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/13 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_non_discard_command. Retrieved 17/20 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 6/12 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/16 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 18/22 statements.


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
    var_11 = [var_10]
    var_12 = [var_2]
    var_13 = lambda x: x * var_6
    var_14 = {var_2: var_6}
    var_15 = {var_5: var_6}
    var_16 = {var_0: var_14, var_1: var_15}

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
    var_6 = 0
    var_7 = {var_4: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'z'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = {var_2: var_4, var_3: var_5}
    var_12 = (var_0, var_11)
    var_13 = [var_12]
    var_14 = [var_2]
    var_15 = {var_3: var_5}
    var_16 = {var_7: var_8}
    var_17 = {var_0: var_15, var_1: var_16}



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'
    var_5 = hasattr(var_3, var_4)



# Parsed testcases at query #37
#--------------------------

# Failed to parse test__get_arity_with_default_parameter.




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
    var_7 = lambda k: k == var_0
    var_8 = module_0._get_keys_and_values(var_6, var_7)



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
    var_7 = 'd'
    var_8 = lambda k: k == var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = 'some'
    var_3 = 'path'
    var_4 = [var_2, var_3]
    var_5 = 'some_command'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'Structure'
    var_1 = ()
    var_2 = 'evolver'
    var_3 = 'Evolver'
    var_4 = ()
    var_5 = {}
    var_6 = []
    var_7 = []
    var_8 = 'discard'



# Parsed testcases at query #42
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'q'
    var_2 = 'r'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




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



# Parsed testcases at query #45
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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__update_structure_with_empty_path_and_discard_command. Retrieved 9/11 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 6/10 statements.
# Partially parsed test__update_structure_with_empty_sentinel_and_discard_command. Retrieved 5/9 statements.
# Partially parsed test__update_structure_with_non_empty_path_and_discard_command. Retrieved 15/17 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []

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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_items_with_non_dict_structure. Retrieved 5/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_get_arity_with_no_args.
# Failed to parse test_get_arity_with_one_arg.
# Failed to parse test_get_arity_with_multiple_args.
# Failed to parse test_get_arity_with_default_args.
# Failed to parse test_get_arity_with_keyword_only_args.
# Failed to parse test_get_arity_with_positional_only_args.




# Parsed testcases at query #50
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



# Parsed testcases at query #51
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v % var_4 == var_3
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
    var_7 = 'b'
    var_8 = module_0._get_keys_and_values(var_6, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []



# Parsed testcases at query #53
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



