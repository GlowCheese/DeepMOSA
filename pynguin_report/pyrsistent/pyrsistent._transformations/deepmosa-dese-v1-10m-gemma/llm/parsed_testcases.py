####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_do_to_path_base_case_with_value. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_base_case_with_command. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_recursive_update_dict. Retrieved 7/15 statements.
# Partially parsed test_do_to_path_recursive_discard. Retrieved 10/18 statements.
# Partially parsed test_do_to_path_with_predicate_in_path. Retrieved 12/16 statements.
# Partially parsed test_do_to_path_with_predicate_arity_two. Retrieved 11/15 statements.
# Partially parsed test_do_to_path_raises_error_on_invalid_arity. Retrieved 6/10 statements.


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
    var_4 = lambda x: x + var_1

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
    var_1 = 'd'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = [var_0, var_2]
    var_9 = {var_3: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'odd'
    var_5 = 'even'
    var_6 = {var_0: var_4, var_1: var_5, var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = lambda k: k % var_1 == var_7
    var_9 = [var_8]
    var_10 = 10
    var_11 = {var_0: var_4, var_1: var_10, var_2: var_4, var_3: var_10}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'odd'
    var_5 = 'even'
    var_6 = {var_0: var_4, var_1: var_5, var_2: var_4, var_3: var_5}
    var_7 = lambda k, v: v == var_5
    var_8 = [var_7]
    var_9 = 0
    var_10 = {var_0: var_4, var_1: var_9, var_2: var_4, var_3: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    var_5 = [var_4]



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = lambda k: k == var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_rex_matches_string. Retrieved 4/6 statements.
# Partially parsed test_rex_does_not_match_string. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 6/9 statements.
# Partially parsed test_rex_regex_special_characters. Retrieved 4/6 statements.
# Partially parsed test_rex_empty_string. Retrieved 4/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'
    var_3 = 'bc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = 'abc'
    var_5 = [var_4]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = ' '



# Parsed testcases at query #4
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only_required.
# Failed to parse test_get_arity_mixed_positional_and_default.
# Failed to parse test_get_arity_ignores_keyword_only.
# Failed to parse test_get_arity_ignores_var_args.




# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = module_0._items(var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = (var_0, var_1)
    var_3 = (var_2,)
    var_4 = module_0._items(var_3)

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_rex_matches_exact_string. Retrieved 2/3 statements.
# Partially parsed test_rex_does_not_match_different_string. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_regex_pattern. Retrieved 4/6 statements.
# Partially parsed test_rex_returns_false_for_non_string_input. Retrieved 5/8 statements.
# Partially parsed test_rex_matches_partial_via_regex_logic. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_empty_string_pattern. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '12a'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'bac'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'any'



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only_required.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_mixed_types.
# Failed to parse test_get_arity_positional_only_explicit.
# Failed to parse test_get_arity_all_required_positional.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_do_to_path_identity. Retrieved 9/12 statements.
# Partially parsed test_do_to_path_no_path_command_value. Retrieved 5/7 statements.
# Partially parsed test_do_to_path_no_path_callable_command. Retrieved 6/8 statements.
# Partially parsed test_do_to_path_single_level_key. Retrieved 8/11 statements.
# Partially parsed test_do_to_path_nested_update. Retrieved 8/15 statements.
# Partially parsed test_do_to_path_with_predicate_unary. Retrieved 14/17 statements.
# Partially parsed test_do_to_path_with_predicate_binary. Retrieved 13/16 statements.
# Partially parsed test_do_to_path_discard_command. Retrieved 7/15 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 7/10 statements.
# Partially parsed test_do_to_path_sequence_indexing. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = lambda x: x + var_2
    var_7 = 'else'
    var_8 = {var_0: var_2, var_7: var_3}

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
    var_4 = 10
    var_5 = lambda x: x[var_0] + var_4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x + var_2
    var_7 = {var_0: var_3, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = lambda x: x + var_2
    var_6 = 2
    var_7 = {var_1: var_6}

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
    var_9 = [var_8]
    var_10 = 10
    var_11 = lambda x: x * var_10
    var_12 = 30
    var_13 = {var_0: var_10, var_1: var_4, var_2: var_12}

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
    var_9 = [var_8]
    var_10 = lambda x: x + var_3
    var_11 = 11
    var_12 = {var_0: var_3, var_1: var_11, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = [var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, extra: var_3
    var_5 = [var_4]
    var_6 = lambda x: x

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_4]
    var_6 = 5
    var_7 = lambda x: x + var_6
    var_8 = 25
    var_9 = [var_0, var_8, var_2]



# Parsed testcases at query #12
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x[var_0] + var_1
    var_5 = module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 2

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'fixed_value'
    var_5 = module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 'fixed_value'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_items_predicate_evaluates_to_false. Retrieved 4/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._items(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_rex_matches_exact_string. Retrieved 2/3 statements.
# Partially parsed test_rex_does_not_match_different_string. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_substring_only_due_to_match_behavior. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_regex_patterns. Retrieved 4/6 statements.
# Partially parsed test_rex_returns_false_for_non_string_input. Retrieved 5/8 statements.
# Partially parsed test_rex_handles_empty_string_and_pattern. Retrieved 3/5 statements.
# Partially parsed test_rex_with_anchors. Retrieved 4/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = 'zabc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = 123
    var_4 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'a'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = 'ab'



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only_no_default.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_mixed_args.
# Failed to parse test_get_arity_ignores_keyword_only.
# Failed to parse test_get_arity_ignores_varargs_and_varkw.
# Failed to parse test_get_arity_positional_only_explicit.




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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._transformations as module_0

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
    var_9 = module_0._get_keys_and_values(var_6, var_8)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda k, v: v == var_2
    var_5 = module_0._get_keys_and_values(var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = lambda x: var_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda k: var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = module_0._get_keys_and_values(var_5, var_1)
    var_7 = len(var_6)



# Parsed testcases at query #19
#--------------------------




import builtins as module_0
import pyrsistent._transformations as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = lambda k: k == var_1
    var_7 = module_1._get_keys_and_values(var_5, var_6)



# Parsed testcases at query #20
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = module_0._items(var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = (var_0, var_1)
    var_3 = module_0._items(var_2)

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



# Parsed testcases at query #21
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'



# Parsed testcases at query #23
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._items(var_3)



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #25
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._transformations as module_0

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
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 1
    var_4 = 5
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 2
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_structure_with_discard. Retrieved 8/12 statements.
# Partially parsed test_update_structure_with_assignment. Retrieved 12/15 statements.
# Partially parsed test_update_structure_with_nested_path. Retrieved 7/15 statements.
# Partially parsed test_update_structure_with_empty_sentinel_expansion. Retrieved 8/10 statements.
# Partially parsed test_update_structure_no_change. Retrieved 6/9 statements.


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
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]
    var_7 = []
    var_8 = 10
    var_9 = lambda x: x + var_8
    var_10 = 11
    var_11 = 12

def test_case_0():
    var_0 = 'old_value'
    var_1 = 'sub_key'
    var_2 = {var_1: var_0}
    var_3 = 'root'
    var_4 = [var_1]
    var_5 = 'new_value'
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = 5
    var_7 = lambda x: var_6

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_1, var_0)
    var_3 = [var_2]
    var_4 = []
    var_5 = lambda x: x



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_get_arity_no_params.
# Failed to parse test_get_arity_positional_only.
# Failed to parse test_get_arity_positional_or_keyword.
# Failed to parse test_get_arity_mixed_types.
# Failed to parse test_get_arity_varargs_and_varkw.
# Failed to parse test_get_arity_all_required_pos_or_kw.




# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_discard_key. Retrieved 9/13 statements.
# Partially parsed test_update_structure_discard_non_existent. Retrieved 8/12 statements.
# Partially parsed test_update_structure_deep_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 7/19 statements.
# Partially parsed test_update_structure_vector_discard_reverse_order. Retrieved 10/14 statements.
# Partially parsed test_update_structure_no_change_if_same_value. Retrieved 7/9 statements.


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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_4]
    var_6 = []
    var_7 = {var_0: var_1}

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
    var_2 = 'c'
    var_3 = [var_2]
    var_4 = 10
    var_5 = lambda x: var_4
    var_6 = {var_2: var_4}

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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'



# Parsed testcases at query #33
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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_update_structure_discard_pmap. Retrieved 10/13 statements.
# Partially parsed test_update_structure_discard_vector. Retrieved 13/16 statements.
# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 8/17 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 9/13 statements.
# Partially parsed test_update_structure_no_change. Retrieved 7/9 statements.


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
    var_12 = [var_1, var_2]

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
    var_5 = 3
    var_6 = lambda x: var_5
    var_7 = {var_0: var_5}

def test_case_0():
    var_0 = {}
    var_1 = 'new_key'
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = 99
    var_7 = lambda x: var_6
    var_8 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._transformations as module_0

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
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 10
    var_4 = 20
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_5
    var_8 = module_0._get_keys_and_values(var_6, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = lambda k: k == var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_structure_path_is_empty_and_command_is_discard. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 9/20 statements.


import builtins as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda e, k: var_0
    var_2 = module_0.object()
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = var_1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 10/13 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = lambda e, k: var_1
    var_3 = 'key1'
    var_4 = 'val1'
    var_5 = (var_3, var_4)
    var_6 = 'key2'
    var_7 = 'val2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_predicate_false_due_to_path. Retrieved 11/15 statements.


import builtins as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x, y: var_0
    var_2 = module_0.object()
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'some'
    var_8 = 'path'
    var_9 = [var_7, var_8]
    var_10 = var_1



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_get_keys_and_values_with_callable_predicate. Retrieved 9/13 statements.


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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'



# Parsed testcases at query #43
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._get_keys_and_values(var_2, var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_items_attribute_error_is_not_raised. Retrieved 6/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._items(var_4)



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda e, k: var_0
    var_2 = ''
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_update_structure_predicate_is_false_by_having_path. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = [var_4]
    var_6 = 'module'
    var_7 = None
    var_8 = '_do_to_path'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_rex_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcdef'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only_required.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_mixed_kinds.
# Failed to parse test_get_arity_positional_only_explicit.
# Failed to parse test_get_arity_all_defaults.




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

import pyrsistent._transformations as module_0

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
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 15
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = lambda k: k == var_3
    var_5 = module_0._get_keys_and_values(var_2, var_4)

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
    var_3 = 'non_existent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #3
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
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
    var_5 = 'a'
    var_6 = module_0._get_keys_and_values(var_4, var_5)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_arity_no_args.
# Failed to parse test_get_arity_positional_only_and_keyword.
# Failed to parse test_get_arity_with_defaults.
# Failed to parse test_get_arity_with_varargs_and_kwargs.
# Failed to parse test_get_arity_positional_only_explicit.
# Failed to parse test_get_arity_keyword_only_ignored.




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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._items(var_0)
    var_3 = list(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_do_to_path_base_case_value. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_base_case_function. Retrieved 6/10 statements.
# Partially parsed test_do_to_path_with_path_and_key. Retrieved 9/13 statements.
# Partially parsed test_do_to_path_with_nested_update. Retrieved 7/15 statements.
# Partially parsed test_do_to_path_with_predicate. Retrieved 10/14 statements.
# Partially parsed test_do_to_path_with_binary_predicate. Retrieved 11/15 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 4/11 statements.
# Partially parsed test_do_to_path_with_discard. Retrieved 7/11 statements.
# Partially parsed test_do_to_path_with_list_structure. Retrieved 6/10 statements.
# Partially parsed test_do_to_path_with_empty_sentinel. Retrieved 6/10 statements.


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
    var_4 = lambda x: x + var_1
    var_5 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x
    var_7 = {var_1: var_2}
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 20
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_1]
    var_8 = 99
    var_9 = {var_0: var_3, var_1: var_8, var_2: var_5}

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
    var_9 = 99
    var_10 = {var_0: var_3, var_1: var_9, var_2: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = {var_1: var_3}

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 1
    var_4 = [var_3]
    var_5 = 99

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = [var_4]



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_do_to_path_leaf_command_value. Retrieved 5/8 statements.
# Partially parsed test_do_to_path_leaf_command_callable. Retrieved 6/10 statements.
# Partially parsed test_do_to_path_with_path_and_key_lookup. Retrieved 8/16 statements.
# Partially parsed test_do_to_path_with_predicate_key. Retrieved 11/17 statements.
# Partially parsed test_do_to_path_error_on_invalid_arity. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 10

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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_1]
    var_5 = 2
    var_6 = 2
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = [var_0]
    var_8 = 10
    var_9 = 10
    var_10 = {var_0: var_9, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = 5
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 5
    var_8 = 2
    var_9 = 3
    var_10 = {var_0: var_7, var_5: var_8, var_6: var_9}



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_rex_matches_valid_string. Retrieved 4/6 statements.
# Partially parsed test_rex_does_not_match_invalid_string. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 6/9 statements.
# Partially parsed test_rex_with_complex_regex. Retrieved 5/8 statements.
# Partially parsed test_rex_empty_regex. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'
    var_3 = 'def'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = 'abc'
    var_5 = [var_4]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}-\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123-456'
    var_3 = '12-345'
    var_4 = 'abc-def'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'anything'



# Parsed testcases at query #16
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x[var_0]
    var_5 = module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_update_structure_discard_mapping. Retrieved 10/15 statements.
# Partially parsed test_update_structure_discard_vector. Retrieved 10/15 statements.
# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_no_change_if_same_value. Retrieved 8/11 statements.
# Partially parsed test_update_structure_discard_non_existent. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = {var_1: var_3}

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = (var_4, var_1)
    var_6 = [var_5]
    var_7 = []
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_0, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = 2
    var_6 = lambda x: var_5
    var_7 = {var_0: var_1}
    var_8 = {var_0: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = [var_1]
    var_5 = 2
    var_6 = lambda x: var_5
    var_7 = {var_1: var_2}
    var_8 = {var_1: var_5}

def test_case_0():
    var_0 = 'new_key'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = lambda x: var_1
    var_6 = {var_0: var_1}
    var_7 = {var_0: var_1}

def test_case_0():
    var_0 = 'non_existent'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_get_arity_predicate_false_due_to_default_value.
# Failed to parse test_get_arity_predicate_false_due_to_parameter_kind.




# Parsed testcases at query #21
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



# Parsed testcases at query #22
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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = module_0._get_keys_and_values(var_2, var_3)

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
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: k == var_1
    var_8 = module_0._get_keys_and_values(var_6, var_7)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = module_0._get_keys_and_values(var_2, var_3)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'nonexistent'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1



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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 0
    var_6 = lambda k: k % var_4 == var_5
    var_7 = module_0._get_keys_and_values(var_3, var_6)

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_false_on_non_string. Retrieved 3/5 statements.
# Partially parsed test_rex_predicate_evaluates_to_false_on_mismatch. Retrieved 3/5 statements.
# Partially parsed test_rex_predicate_evaluates_to_false_on_none. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'def'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_update_structure_predicate_false_when_path_is_not_empty. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = (var_4,)
    var_6 = 'discard'
    var_7 = 'mock'



# Parsed testcases at query #26
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_rex_matches_valid_string. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_invalid_string. Retrieved 3/4 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 3/4 statements.
# Partially parsed test_rex_handles_none_input. Retrieved 3/4 statements.
# Partially parsed test_rex_exact_match. Retrieved 4/6 statements.
# Partially parsed test_rex_case_sensitivity. Retrieved 4/6 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcde'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'xyz'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 123

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = None

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[A-Z]$'
    var_1 = module_0.rex(var_0)
    var_2 = 'A'
    var_3 = 'a'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_update_structure_discard_single_key. Retrieved 7/13 statements.
# Partially parsed test_update_structure_discard_multiple_keys. Retrieved 10/16 statements.
# Partially parsed test_update_structure_set_value. Retrieved 10/15 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 9/19 statements.
# Partially parsed test_update_structure_empty_sentinel_expansion. Retrieved 8/11 statements.
# Partially parsed test_update_structure_complex_path_with_discard. Retrieved 24/37 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = {var_0: var_1}

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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = {var_0: var_1}
    var_9 = {var_0: var_6}

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
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = 10
    var_6 = lambda x: var_5
    var_7 = {var_0: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = (var_4, var_0)
    var_6 = (var_1, var_2)
    var_7 = [var_5, var_6]
    var_8 = []
    var_9 = 'indices'
    var_10 = [var_0, var_1, var_2]
    var_11 = (var_4, var_0)
    var_12 = (var_1, var_2)
    var_13 = [var_11, var_12]
    var_14 = [var_0, var_1, var_2]
    var_15 = []
    var_16 = lambda x: x
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_0, var_18: var_1}
    var_20 = (var_17, var_0)
    var_21 = (var_18, var_1)
    var_22 = [var_20, var_21]
    var_23 = []



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_structure_path_is_empty_and_command_is_discard. Retrieved 11/15 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda x, y: var_0
    var_2 = 'key1'
    var_3 = 'val1'
    var_4 = (var_2, var_3)
    var_5 = 'key2'
    var_6 = 'val2'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = None
    var_10 = var_1



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_rex_matches_valid_string. Retrieved 4/6 statements.
# Partially parsed test_rex_does_not_match_invalid_string. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 6/9 statements.
# Partially parsed test_rex_matches_exact_string. Retrieved 5/8 statements.
# Partially parsed test_rex_regex_special_characters. Retrieved 5/8 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'def'
    var_3 = 'ab'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = 'abc'
    var_5 = [var_4]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello '
    var_4 = 'hello world'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d{3}-\\d{3}'
    var_1 = module_0.rex(var_0)
    var_2 = '123-456'
    var_3 = '12-345'
    var_4 = 'abc-def'



# Parsed testcases at query #32
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_keys_and_values_with_binary_predicate. Retrieved 8/9 statements.


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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = module_0._get_keys_and_values(var_2, var_3)

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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = lambda k, v: v > var_3

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_rex_matches_exact_string. Retrieved 2/3 statements.
# Partially parsed test_rex_does_not_match_different_string. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_substring_if_not_anchored. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_middle_of_string. Retrieved 3/4 statements.
# Partially parsed test_rex_handles_regex_patterns. Retrieved 4/6 statements.
# Partially parsed test_rex_returns_false_for_non_string_types. Retrieved 5/8 statements.
# Partially parsed test_rex_handles_empty_string_and_pattern. Retrieved 3/5 statements.
# Partially parsed test_rex_case_sensitivity. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'zabc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'a'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'ABC'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_update_structure_discard_mapping. Retrieved 10/14 statements.
# Partially parsed test_update_structure_replace_value. Retrieved 9/12 statements.
# Partially parsed test_update_structure_nested_update. Retrieved 8/16 statements.
# Partially parsed test_update_structure_expansion_with_empty_sentinel. Retrieved 10/13 statements.
# Partially parsed test_update_structure_no_change_if_value_same. Retrieved 7/10 statements.


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
    var_5 = []
    var_6 = 2
    var_7 = lambda x: var_6
    var_8 = var_7

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_1]
    var_6 = 20
    var_7 = lambda x: var_6

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = (var_4, var_0)
    var_6 = [var_5]
    var_7 = []
    var_8 = 5
    var_9 = lambda x: var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = (var_0, var_1)
    var_4 = [var_3]
    var_5 = []
    var_6 = lambda x: var_1



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_rex_matches_exact_string. Retrieved 2/3 statements.
# Partially parsed test_rex_does_not_match_different_string. Retrieved 3/4 statements.
# Partially parsed test_rex_does_not_match_substring_without_anchor. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_with_anchors. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 5/8 statements.
# Partially parsed test_rex_handles_regex_patterns. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_empty_string_and_pattern. Retrieved 5/7 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'anything'
    var_3 = 'abc'
    var_4 = module_0.rex(var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/16 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 5/20 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda e, k: var_0
    var_2 = []
    var_3 = []
    var_4 = var_1



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_structure_predicate_false_when_path_exists. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = (var_4,)
    var_6 = 'discard'
    var_7 = 'mock_module'



# Parsed testcases at query #41
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

import pyrsistent._transformations as module_0

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
    var_9 = module_0._get_keys_and_values(var_6, var_8)

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

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'zero'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_get_keys_and_values_raises_value_error_on_invalid_arity. Retrieved 3/9 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._get_keys_and_values(var_4, var_0)

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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'a'
    var_8 = lambda k: k.startswith(var_7)
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 7
    var_5 = lambda x: x > var_4
    var_6 = module_0._get_keys_and_values(var_3, var_5)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 2
    var_8 = lambda k, v: v > var_7
    var_9 = module_0._get_keys_and_values(var_6, var_8)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda i, v: i == var_4 and v == var_1
    var_6 = module_0._get_keys_and_values(var_3, var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_rex_predicate_evaluates_to_true. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_update_structure_predicate_false_by_path_exists. Retrieved 11/28 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'some_path'
    var_5 = (var_4,)
    var_6 = []
    var_7 = 'not_empty'
    var_8 = (var_7,)
    var_9 = 'exists'
    var_10 = (var_9,)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_update_structure_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #46
#--------------------------




import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = module_0._get_keys_and_values(var_2, var_3)



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_rex_matches_exact_string. Retrieved 2/3 statements.
# Partially parsed test_rex_does_not_match_different_string. Retrieved 3/4 statements.
# Partially parsed test_rex_matches_regex_pattern. Retrieved 4/6 statements.
# Partially parsed test_rex_handles_non_string_input. Retrieved 5/8 statements.
# Partially parsed test_rex_partial_match_fails_due_to_match_behavior. Retrieved 4/6 statements.
# Partially parsed test_rex_empty_string_pattern. Retrieved 3/5 statements.


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'def'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '12a'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = 123
    var_4 = [var_0]

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = 'zabc'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = 'anything'



