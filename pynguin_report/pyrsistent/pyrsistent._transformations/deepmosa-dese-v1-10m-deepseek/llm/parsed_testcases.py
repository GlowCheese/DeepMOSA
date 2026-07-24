####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__do_to_path_with_discard_command. Retrieved 8/10 statements.


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
    var_7 = [var_5, var_6]
    var_8 = module_0._do_to_path(var_3, var_4, var_7)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 5
    var_4 = 10
    var_5 = 15
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 7
    var_8 = lambda k, v: v > var_7
    var_9 = [var_8]
    var_10 = 2
    var_11 = lambda x: x * var_10
    var_12 = module_0._do_to_path(var_6, var_9, var_11)

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
    var_11 = [var_0, var_2]
    var_12 = 10
    var_13 = lambda x: x + var_12
    var_14 = module_0._do_to_path(var_10, var_11, var_13)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_4]
    var_6 = 3
    var_7 = lambda x: x * var_6
    var_8 = module_0._do_to_path(var_3, var_5, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = lambda k: k == var_4
    var_6 = [var_5]
    var_7 = lambda x: x - var_0
    var_8 = module_0._do_to_path(var_3, var_6, var_7)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_rex_matches_correct_string. Retrieved 4/7 statements.
# Partially parsed test_rex_does_not_match_incorrect_string. Retrieved 3/5 statements.
# Partially parsed test_rex_returns_none_for_non_string_input. Retrieved 5/9 statements.
# Partially parsed test_rex_uses_fullmatch_behavior_with_end_anchor. Retrieved 4/7 statements.
# Partially parsed test_rex_pattern_with_special_characters. Retrieved 4/7 statements.
# Partially parsed test_rex_case_sensitive_by_default. Retrieved 4/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello world'
    var_3 = 'hello'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'world hello'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^exact$'
    var_1 = module_0.rex(var_0)
    var_2 = 'exact'
    var_3 = 'exact extra'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+\\.\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123.456'
    var_3 = 'abc.def'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^CaseSensitive$'
    var_1 = module_0.rex(var_0)
    var_2 = 'CaseSensitive'
    var_3 = 'casesensitive'



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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)

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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0._get_keys_and_values(var_3, var_4)



# Parsed testcases at query #4
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




# Parsed testcases at query #5
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'ab'
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 3/59 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2



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
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_false. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'b'



# Parsed testcases at query #10
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = enumerate(var_3)
    var_6 = list(var_5)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_var_positional_parameter.
# Failed to parse test_get_arity_with_var_keyword_parameter.
# Failed to parse test_get_arity_with_mixed_parameter_types.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_callable_command. Retrieved 10/21 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_multiple_kvs_and_discard_command. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_nested_path_and_callable_command. Retrieved 11/28 statements.
# Partially parsed test_update_structure_with_no_change_in_value. Retrieved 7/10 statements.
# Partially parsed test_update_structure_with_empty_kvs. Retrieved 7/10 statements.


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
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 2
    var_7 = lambda v: v * var_6
    var_8 = 20
    var_9 = {var_1: var_8}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = []
    var_5 = 100
    var_6 = lambda v: var_5
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
    var_2 = 'c'
    var_3 = 5
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = [var_1, var_2]
    var_7 = 10
    var_8 = lambda v: v + var_7
    var_9 = 15
    var_10 = {var_2: var_9}

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
    var_3 = []
    var_4 = []
    var_5 = 99
    var_6 = lambda v: var_5



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #14
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/53 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #16
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #17
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = enumerate(var_3)
    var_6 = list(var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__get_keys_and_values_with_non_existent_non_callable_key_in_mapping. Retrieved 6/8 statements.
# Partially parsed test__get_keys_and_values_with_out_of_range_non_callable_key_in_sequence. Retrieved 6/8 statements.


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
    var_4 = lambda k, v, x: var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

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



# Parsed testcases at query #19
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #20
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'param2'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 10/17 statements.
# Partially parsed test_update_structure_with_non_empty_path_and_discard_command. Retrieved 7/21 statements.
# Partially parsed test_update_structure_with_empty_path_and_non_discard_command. Retrieved 9/16 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard_command. Retrieved 6/19 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_non_discard_command. Retrieved 10/24 statements.


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
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_0: var_1, var_3: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = [var_1]
    var_5 = {var_1: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = 'y'
    var_6 = [var_5]
    var_7 = {var_1: var_2}
    var_8 = None
    var_9 = {var_5: var_8}



# Parsed testcases at query #24
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._items(var_3)
    var_5 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_rex_returns_lambda_for_string_matching. Retrieved 6/10 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'other_string'
    var_5 = 123



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_error. Retrieved 1/6 statements.
# Partially parsed test_predicate_with_three_args_raises_error. Retrieved 1/6 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = {}



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
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]



# Parsed testcases at query #28
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)



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
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]

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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_update_structure_with_path_and_command_not_discard. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'x'
    var_8 = [var_7]
    var_9 = lambda x: x



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = [var_5]
    var_7 = lambda x: x
    var_8 = {var_0: var_1}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = 6
    var_7 = (var_5, var_6)
    var_8 = []
    var_9 = {var_1: var_3}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = [var_6]
    var_8 = lambda x: x
    var_9 = {var_0: var_1}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 10/17 statements.


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



# Parsed testcases at query #35
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_items_without_attribute_error_returns_items.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'b'



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
    var_10 = (var_0, var_3)
    var_11 = (var_2, var_5)
    var_12 = [var_10, var_11]

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]

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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some'
    var_5 = 'path'
    var_6 = [var_4, var_5]
    var_7 = lambda x: x



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_with_empty_path_and_discard_command. Retrieved 9/30 statements.


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



# Parsed testcases at query #41
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
    var_4 = lambda : var_3
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
    var_4 = 0
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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 0



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
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 23/66 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'nested'
    var_5 = 'path'
    var_6 = lambda x: x
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = 0
    var_12 = 10
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = lambda x: x
    var_16 = 'a'
    var_17 = 'b'
    var_18 = lambda x: x
    var_19 = []
    var_20 = [var_7, var_8]
    var_21 = lambda x: x
    var_22 = 'x'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_discard_command_and_nested_path. Retrieved 9/21 statements.
# Partially parsed test_update_structure_with_discard_command_and_multiple_kvs. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_callable_command_and_nested_path. Retrieved 9/20 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_discard_command. Retrieved 6/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_callable_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_nested_path. Retrieved 9/17 statements.
# Partially parsed test_update_structure_with_unchanged_value_and_non_empty_path. Retrieved 8/19 statements.
# Partially parsed test_update_structure_with_unchanged_value_and_empty_path. Retrieved 8/12 statements.


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
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = {var_2: var_4}

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
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x + var_1
    var_7 = 2
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = 10
    var_7 = lambda x: var_6
    var_8 = {var_4: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
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
    var_6 = lambda x: x
    var_7 = {var_0: var_1}



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_var_positional_parameters.
# Failed to parse test_get_arity_with_var_keyword_parameters.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_mixed_parameters.




# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_value_error. Retrieved 1/6 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_discard_command_and_non_empty_path. Retrieved 9/20 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_non_discard_command. Retrieved 8/16 statements.
# Partially parsed test_update_structure_with_empty_sentinel_value_and_discard_command. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_multiple_kvs_and_discard_command. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_nested_structure_and_path. Retrieved 11/28 statements.
# Partially parsed test_update_structure_with_result_equal_to_original_value. Retrieved 8/12 statements.
# Partially parsed test_update_structure_with_empty_kvs_list. Retrieved 8/12 statements.


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
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_3}
    var_7 = [var_2]
    var_8 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = []
    var_8 = {var_0: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 100
    var_6 = []
    var_7 = 100
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = []
    var_7 = {}

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
    var_8 = (var_2, var_5)
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = {var_1: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 'i'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = 2
    var_7 = lambda x: x * var_6
    var_8 = [var_1, var_2]
    var_9 = 20
    var_10 = {var_2: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = lambda x: x
    var_6 = []
    var_7 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = []
    var_7 = {var_0: var_1}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'nested'
    var_5 = [var_4]
    var_6 = lambda x: x



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_with_arity_0_evaluates_to_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_discard_command_and_nested_path. Retrieved 9/20 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_discard_command_and_sentinel_value. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_command_and_sentinel_value_creating_new_node. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_multiple_kvs_and_discard_reversed. Retrieved 12/16 statements.
# Partially parsed test_update_structure_with_nested_structure_and_path. Retrieved 11/28 statements.
# Partially parsed test_update_structure_with_no_change_returns_same_structure. Retrieved 7/10 statements.
# Partially parsed test_update_structure_with_empty_kvs_returns_unchanged_structure. Retrieved 5/9 statements.


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
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = []
    var_8 = {var_0: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 100
    var_6 = []
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
    var_4 = 5
    var_5 = []
    var_6 = 5
    var_7 = {var_0: var_1, var_3: var_6}

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
    var_2 = 'c'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = 5
    var_7 = lambda x: x + var_6
    var_8 = [var_1, var_2]
    var_9 = 15
    var_10 = {var_2: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = lambda x: x
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = []



# Parsed testcases at query #54
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda i: i == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = (var_4, var_1)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda i, v: v == var_2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 2
    var_7 = (var_6, var_2)
    var_8 = [var_7]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 4/12 statements.
# Partially parsed test_update_structure_discard_nested. Retrieved 5/14 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_nonexistent_key. Retrieved 3/9 statements.
# Partially parsed test_update_structure_set_leaf. Retrieved 6/10 statements.
# Partially parsed test_update_structure_set_nested. Retrieved 5/14 statements.
# Partially parsed test_update_structure_set_new_empty_node. Retrieved 4/11 statements.
# Partially parsed test_update_structure_set_with_callable. Retrieved 6/12 statements.
# Partially parsed test_update_structure_discard_from_vector. Retrieved 11/15 statements.
# Partially parsed test_update_structure_discard_specific_index_from_vector. Retrieved 7/11 statements.
# Partially parsed test_update_structure_no_change_when_value_unchanged. Retrieved 6/9 statements.
# Partially parsed test_update_structure_empty_path_and_command. Retrieved 5/9 statements.
# Partially parsed test_update_structure_with_sentinel_and_discard. Retrieved 3/9 statements.


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
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = 42

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'x'
    var_3 = [var_2]
    var_4 = 99

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = [var_1]
    var_3 = 100

def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = 6

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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 1
    var_4 = (var_3, var_1)
    var_5 = [var_4]
    var_6 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = [var_3]
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_arity_positional_only. Retrieved 1/4 statements.
# Partially parsed test_get_arity_with_defaults. Retrieved 1/4 statements.
# Partially parsed test_get_arity_keyword_only. Retrieved 1/4 statements.
# Partially parsed test_get_arity_var_positional. Retrieved 1/4 statements.
# Partially parsed test_get_arity_mixed. Retrieved 1/4 statements.
# Partially parsed test_get_arity_no_parameters. Retrieved 1/4 statements.
# Partially parsed test_get_arity_positional_or_keyword. Retrieved 1/4 statements.
# Partially parsed test_get_arity_builtin. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 3

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 1



# Parsed testcases at query #2
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = []

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__do_to_path_with_single_key_path_and_callable_command. Retrieved 8/12 statements.
# Partially parsed test__do_to_path_with_single_key_path_and_non_callable_command. Retrieved 6/10 statements.
# Partially parsed test__do_to_path_with_nested_path_and_callable_command. Retrieved 9/17 statements.
# Partially parsed test__do_to_path_with_callable_key_spec_unary. Retrieved 10/14 statements.
# Partially parsed test__do_to_path_with_callable_key_spec_binary. Retrieved 10/14 statements.
# Partially parsed test__do_to_path_with_discard_command_on_single_key. Retrieved 7/11 statements.
# Partially parsed test__do_to_path_with_discard_command_on_multiple_keys_via_callable. Retrieved 11/15 statements.
# Partially parsed test__do_to_path_with_nonexistent_key_and_non_callable_command. Retrieved 7/11 statements.
# Partially parsed test__do_to_path_with_nonexistent_key_and_discard_command. Retrieved 6/10 statements.
# Partially parsed test__do_to_path_with_list_structure_and_index_path. Retrieved 10/14 statements.
# Partially parsed test__do_to_path_with_list_structure_and_callable_index. Retrieved 10/14 statements.
# Partially parsed test__do_to_path_with_discard_on_list_structure. Retrieved 7/11 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x * var_3
    var_7 = {var_0: var_3, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = 100
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 1
    var_6 = lambda x: x + var_5
    var_7 = 6
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k: k == var_0
    var_6 = [var_5]
    var_7 = 10
    var_8 = lambda v: v * var_7
    var_9 = {var_0: var_7, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda k, v: v > var_2
    var_6 = [var_5]
    var_7 = lambda v: v * var_3
    var_8 = 4
    var_9 = {var_0: var_2, var_1: var_8}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = {var_1: var_3}

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
    var_9 = [var_8]
    var_10 = {var_0: var_3, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 99
    var_6 = {var_0: var_1, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_4]
    var_6 = 2
    var_7 = lambda x: x * var_6
    var_8 = 40
    var_9 = [var_0, var_8, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = lambda i: i == var_4
    var_6 = [var_5]
    var_7 = lambda x: x + var_0
    var_8 = 30
    var_9 = [var_0, var_1, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = [var_0]
    var_6 = [var_0, var_2, var_3]



# Parsed testcases at query #4
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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_rex_returns_lambda. Retrieved 6/10 statements.
# Partially parsed test_rex_matches_correct_pattern. Retrieved 5/9 statements.
# Partially parsed test_rex_with_special_characters. Retrieved 5/9 statements.
# Partially parsed test_rex_empty_string. Retrieved 4/7 statements.
# Partially parsed test_rex_case_sensitive. Retrieved 5/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'other_string'
    var_5 = 123

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a'
    var_4 = 'b123'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+\\.\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '3.14'
    var_3 = 'abc'
    var_4 = '123'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = 'a'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^Hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'Hello'
    var_3 = 'hello'
    var_4 = 'HELLO'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 4/12 statements.
# Partially parsed test_update_structure_discard_nested. Retrieved 5/14 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_non_existing_key. Retrieved 3/11 statements.
# Partially parsed test_update_structure_set_leaf. Retrieved 4/15 statements.
# Partially parsed test_update_structure_set_nested. Retrieved 5/19 statements.
# Partially parsed test_update_structure_create_new_key. Retrieved 4/15 statements.
# Partially parsed test_update_structure_create_nested_new_key. Retrieved 6/15 statements.
# Partially parsed test_update_structure_with_vector. Retrieved 7/20 statements.
# Partially parsed test_update_structure_discard_vector_reverse. Retrieved 9/13 statements.


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
    var_0 = 1
    var_1 = 'a'
    var_2 = []
    var_3 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'x'
    var_3 = [var_2]
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = []
    var_3 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = 'y'
    var_3 = [var_2]
    var_4 = 2
    var_5 = lambda s: var_4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 0
    var_3 = 'x'
    var_4 = [var_3]
    var_5 = lambda s: s * var_1
    var_6 = 4

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



# Parsed testcases at query #7
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__get_keys_and_values_with_object_and_non_callable_key. Retrieved 2/5 statements.
# Partially parsed test__get_keys_and_values_with_object_and_non_existent_non_callable_key_returns_sentinel. Retrieved 2/5 statements.


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
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda k, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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

def test_case_0():
    var_0 = 5
    var_1 = 'x'

def test_case_0():
    var_0 = 5
    var_1 = 'y'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only.
# Failed to parse test_get_arity_with_positional_or_keyword.
# Failed to parse test_get_arity_with_keyword_only.
# Failed to parse test_get_arity_with_var_positional.
# Failed to parse test_get_arity_with_var_keyword.
# Failed to parse test_get_arity_with_default_values.
# Failed to parse test_get_arity_mixed_parameters.
# Failed to parse test_get_arity_on_builtin.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 2



# Parsed testcases at query #10
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #11
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_arity_other_than_1_or_2_raises_value_error. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_arity_other_than_1_or_2_raises_value_error. Retrieved 1/5 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_existent_non_callable_key_in_mapping. Retrieved 6/8 statements.
# Partially parsed test_get_keys_and_values_with_out_of_range_non_callable_key_in_sequence. Retrieved 6/8 statements.


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
    var_5 = 'b'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 5



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 2/39 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = [var_5]
    var_7 = lambda x: x
    var_8 = {var_0: var_1}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'c'



# Parsed testcases at query #18
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
    var_12 = list(var_4)

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
    var_12 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = []
    var_3 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = []
    var_3 = list(var_1)

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
    var_9 = list(var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_arity_other_than_one_or_two_raises_error. Retrieved 3/7 statements.
# Partially parsed test_predicate_with_three_args_raises_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_get_arity_with_no_parameters.
# Failed to parse test_get_arity_with_positional_only_parameters.
# Failed to parse test_get_arity_with_positional_or_keyword_parameters.
# Failed to parse test_get_arity_with_keyword_only_parameters.
# Failed to parse test_get_arity_with_var_positional_parameter.
# Failed to parse test_get_arity_with_var_keyword_parameter.
# Failed to parse test_get_arity_with_default_parameters.
# Failed to parse test_get_arity_with_mixed_parameters.
# Failed to parse test_get_arity_with_class_method.
# Failed to parse test_get_arity_with_static_method.
# Failed to parse test_get_arity_with_builtin_function.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 2



# Parsed testcases at query #23
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
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda i: i % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = lambda i, v: v > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k: var_5
    var_7 = module_0._get_keys_and_values(var_4, var_6)



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
    var_7 = lambda k: k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)
    var_9 = (var_1, var_4)
    var_10 = [var_9]

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
    var_9 = (var_1, var_4)
    var_10 = [var_9]

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
    var_7 = lambda k: k == var_1
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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)



# Parsed testcases at query #26
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda k, v: v > var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = 3
    var_7 = {var_2: var_0, var_3: var_5, var_4: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_1)
    var_9 = (var_3, var_5)
    var_10 = (var_4, var_6)
    var_11 = [var_9, var_10]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = module_0._get_keys_and_values(var_3, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda i: i % var_0 == var_1
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = 40
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_1, var_3)
    var_10 = (var_0, var_5)
    var_11 = [var_9, var_10]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 25
    var_1 = lambda i, v: v > var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = 40
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0._get_keys_and_values(var_6, var_1)
    var_8 = 2
    var_9 = (var_8, var_4)
    var_10 = 3
    var_11 = (var_10, var_5)
    var_12 = [var_9, var_11]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_rex_function_matches_correct_string. Retrieved 4/7 statements.
# Partially parsed test_rex_function_returns_none_for_non_string. Retrieved 5/8 statements.
# Partially parsed test_rex_function_uses_fullmatch_behavior. Retrieved 3/6 statements.
# Partially parsed test_rex_function_with_special_regex_chars. Retrieved 4/7 statements.
# Partially parsed test_rex_function_case_sensitive. Retrieved 3/6 statements.
# Partially parsed test_rex_function_with_empty_string. Retrieved 3/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc123'
    var_3 = 'xyz'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = 'list'
    var_4 = [var_3]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello world'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'a'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_rex_returns_lambda_for_string_matching. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_string'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_arity_0. Retrieved 3/7 statements.
# Partially parsed test_get_keys_and_values_with_callable_arity_3. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = [var_4]
    var_6 = lambda x: x



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_rex_returns_true_for_matching_string. Retrieved 3/4 statements.
# Partially parsed test_rex_returns_false_for_non_matching_string. Retrieved 3/4 statements.
# Partially parsed test_rex_returns_false_for_non_string_key. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_exact_string. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_partial_exact_string. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_empty_string. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_with_dot_wildcard. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_with_dot_wildcard_wrong_length. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_character_class. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_character_class. Retrieved 3/4 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_string'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'no_match'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^exact$'
    var_1 = module_0.rex(var_0)
    var_2 = 'exact'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^exact$'
    var_1 = module_0.rex(var_0)
    var_2 = 'exact_extra'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^t.st$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^t.st$'
    var_1 = module_0.rex(var_0)
    var_2 = 'tesst'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = module_0._get_keys_and_values(var_4, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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



# Parsed testcases at query #33
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'b'
    var_2 = lambda k, v: k > var_0 and v.startswith(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 'bcd'
    var_7 = {var_0: var_5, var_3: var_1, var_4: var_6}
    var_8 = module_0._get_keys_and_values(var_7, var_2)
    var_9 = (var_3, var_1)
    var_10 = (var_4, var_6)
    var_11 = [var_9, var_10]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 42
    var_4 = 100
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = 'a'
    var_4 = {var_2: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda a, b, c: var_0
    var_2 = 0
    var_3 = 'a'
    var_4 = {var_2: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda k: var_0
    var_2 = {}
    var_3 = module_0._get_keys_and_values(var_2, var_1)
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda k, v: var_0
    var_2 = {}
    var_3 = module_0._get_keys_and_values(var_2, var_1)
    var_4 = []



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_non_callable_key_spec_with_missing_key_returns_sentinel. Retrieved 6/8 statements.


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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    var_4 = []

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = 'b'



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
    var_7 = [var_0, var_2]
    var_8 = lambda k: k in var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)
    var_10 = sorted(var_9)

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
    var_4 = lambda k, v: v == var_2
    var_5 = module_0._get_keys_and_values(var_3, var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)

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
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda k, v: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_discard_with_empty_path. Retrieved 10/14 statements.
# Partially parsed test_update_structure_discard_specific_key. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_non_existent_key. Retrieved 6/12 statements.
# Partially parsed test_update_structure_update_leaf_with_callable. Retrieved 9/13 statements.
# Partially parsed test_update_structure_update_leaf_with_value. Retrieved 9/13 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/20 statements.
# Partially parsed test_update_structure_create_new_nested_node. Retrieved 8/19 statements.
# Partially parsed test_update_structure_discard_nested_key. Retrieved 9/20 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_command. Retrieved 6/12 statements.
# Partially parsed test_update_structure_with_empty_sentinel_and_discard. Retrieved 4/10 statements.
# Partially parsed test_update_structure_multiple_kvs_update. Retrieved 12/16 statements.
# Partially parsed test_update_structure_multiple_kvs_discard_reversed. Retrieved 10/14 statements.
# Partially parsed test_update_structure_no_change_when_result_equals_value. Retrieved 7/10 statements.


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
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_1: var_3}

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
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = lambda x: x + var_1
    var_6 = []
    var_7 = 2
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = 100
    var_6 = []
    var_7 = 100
    var_8 = {var_0: var_7}

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
    var_1 = {}
    var_2 = {}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 5
    var_6 = 5
    var_7 = {var_3: var_6}

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

def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 10
    var_3 = []
    var_4 = 10
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = (var_1, var_3)
    var_7 = [var_5, var_6]
    var_8 = lambda x: x * var_3
    var_9 = []
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
    var_9 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = lambda x: x
    var_6 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = {}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_discard_leaf. Retrieved 7/18 statements.
# Partially parsed test_update_structure_discard_nonexistent. Retrieved 6/17 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 9/20 statements.
# Partially parsed test_update_structure_update_leaf. Retrieved 9/20 statements.
# Partially parsed test_update_structure_update_empty_sentinel. Retrieved 8/19 statements.
# Partially parsed test_update_structure_discard_empty_sentinel. Retrieved 6/17 statements.
# Partially parsed test_update_structure_multiple_kvs. Retrieved 14/29 statements.
# Partially parsed test_update_structure_no_path_no_command. Retrieved 9/13 statements.
# Partially parsed test_update_structure_no_path_callable_command. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_reverse_order. Retrieved 12/27 statements.


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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_1]
    var_8 = {var_2: var_4}

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
    var_1 = {}
    var_2 = {}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = {var_2: var_3}
    var_8 = {var_2: var_5}
    var_9 = [var_2]
    var_10 = lambda x: x * var_5
    var_11 = {var_2: var_5}
    var_12 = 4
    var_13 = {var_2: var_12}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 2
    var_7 = 2
    var_8 = {var_0: var_7}

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
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 0
    var_7 = [var_0, var_1]
    var_8 = [var_3, var_4]
    var_9 = [var_6]
    var_10 = [var_1]
    var_11 = [var_4]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_with_discard_command_and_empty_path. Retrieved 9/14 statements.
# Partially parsed test_update_structure_with_discard_command_and_non_existent_key. Retrieved 7/13 statements.
# Partially parsed test_update_structure_with_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_non_callable_command_and_empty_path. Retrieved 9/13 statements.
# Partially parsed test_update_structure_with_nested_path_and_command. Retrieved 9/20 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_discard. Retrieved 5/11 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_non_discard_command. Retrieved 8/14 statements.
# Partially parsed test_update_structure_with_sentinel_value_and_nested_path. Retrieved 9/17 statements.
# Partially parsed test_update_structure_with_multiple_kvs_and_discard_reversed. Retrieved 12/17 statements.
# Partially parsed test_update_structure_with_evolver_persistent. Retrieved 7/10 statements.


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
    var_5 = 'c'
    var_6 = []

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
    var_3 = 'b'
    var_4 = []

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_4]
    var_6 = 10
    var_7 = 10
    var_8 = {var_4: var_7}

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
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: x



# Parsed testcases at query #41
#--------------------------

# Partially parsed test__get_keys_and_values_with_non_existent_non_callable_key_returns_empty_sentinel. Retrieved 6/8 statements.
# Partially parsed test__get_keys_and_values_with_empty_structure_and_non_callable. Retrieved 4/6 statements.


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
    var_5 = 'b'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda k: var_1
    var_3 = module_0._get_keys_and_values(var_0, var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'missing'
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = 'missing'



# Parsed testcases at query #42
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_0)
    var_7 = (var_1, var_3)
    var_8 = [var_7]



# Parsed testcases at query #43
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
    var_12 = list(var_4)

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
    var_12 = list(var_4)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'only'
    var_1 = [var_0]
    var_2 = module_0._items(var_1)
    var_3 = list(var_2)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_get_keys_and_values_with_non_callable_key_missing. Retrieved 8/10 statements.
# Partially parsed test_get_keys_and_values_with_sequence_structure_non_callable_missing. Retrieved 7/9 statements.
# Partially parsed test_get_keys_and_values_with_object_structure. Retrieved 5/11 statements.
# Partially parsed test_get_keys_and_values_with_object_structure_missing. Retrieved 2/9 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = (var_0, var_2)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = module_0._get_keys_and_values(var_4, var_5)
    var_7 = 'c'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)
    var_7 = (var_4, var_1)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 2
    var_7 = (var_6, var_2)
    var_8 = [var_7]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = 5

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = [var_3]

def test_case_0():
    var_0 = 'z'
    var_1 = 'z'



