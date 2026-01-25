####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_romanize_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_invalid_locale.
# Partially parsed test_romanize_invalid_input_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_join_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_non_string_items. Retrieved 6/7 statements.
# Partially parsed test_join_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_non_iterable_raises_typeerror. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = [var_1]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_value. Retrieved 3/4 statements.
# Partially parsed test_redact_with_zero_value. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CUSTOM]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 0

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_apply_if_transform_applied_when_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_no_transform_when_condition_false. Retrieved 6/7 statements.
# Partially parsed test_apply_if_otherwise_transform_applied_when_condition_false. Retrieved 8/9 statements.
# Partially parsed test_apply_if_string_transformation. Retrieved 3/7 statements.
# Partially parsed test_apply_if_string_no_transformation. Retrieved 3/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 10
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 3

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'test'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_non_iterable_raises_type_error. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_wrap_default_before_and_after. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_non_string_raises_typeerror. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_only_before. Retrieved 3/4 statements.
# Partially parsed test_wrap_with_only_after. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '{'
    var_1 = '}'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = module_0.wrap(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ')'
    var_1 = module_0.wrap(after=var_0)
    var_2 = 'test'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_truncate_returns_original_string_when_shorter_than_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_returns_truncated_string_when_longer_than_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_uses_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'longstring'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'longstring'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_romanize_raises_error_for_unsupported_locale. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'en'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_truncate_with_long_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_short_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_exact_length_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Exactly10'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'This is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 12345



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_prefix_handles_empty_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_with_empty_prefix. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_truncate_with_short_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_long_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_with_exact_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_raises_type_error. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'this is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '[more]'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'this is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'exact'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_prefix_predicate_evaluates_to_false. Retrieved 4/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'user_order'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #16
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/3 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 2/6 statements.
# Partially parsed test_pipe_mixed_functions_with_and_without_random. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 4

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 10

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 2



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_join_predicate_evaluates_to_true. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_is_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_either_value_with_probability. Retrieved 6/9 statements.
# Partially parsed test_maybe_handles_non_string_values. Retrieved 5/6 statements.
# Partially parsed test_maybe_handles_none_values. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = set()
    var_5 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 0

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = None
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'not_none'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_suffix_adds_suffix_to_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_suffix_handles_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_handles_empty_suffix. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pipe_with_multiple_functions. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test-'
    var_1 = 'hello'
    var_2 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_wrap_function_returns_correct_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_function_raises_type_error_for_non_string_input. Retrieved 4/6 statements.
# Partially parsed test_wrap_function_with_default_values. Retrieved 2/3 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'dynamics'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'dynamics'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pipe_function_with_multiple_functions. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'pre-'
    var_1 = '-suf'
    var_2 = 'test'



# Parsed testcases at query #27
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_romanize_with_invalid_locale.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_number_input. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'sensitive data'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'CENSORED'
    var_1 = module_0.redact(var_0)
    var_2 = 'secret info'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[HIDDEN]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'XXX'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[NUMBER]'
    var_1 = module_0.redact(var_0)
    var_2 = 12345



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_number_input. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'secret'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[HIDDEN]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[EMPTY]'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[NUMBER]'
    var_1 = module_0.redact(var_0)
    var_2 = 12345



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_maybe_closure_returns_original_value_when_probability_is_zero. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 'other_value'
    var_3 = 0.0
    var_4 = module_1.maybe(var_2, var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_maybe_returns_original_value_when_probability_out_of_range. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'original'
    var_2 = 'test'
    var_3 = 0
    var_4 = module_1.maybe(var_2, var_3)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 3/5 statements.
# Partially parsed test_romanize_with_kazakh_locale. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_ukrainian_locale. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'kk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/5 statements.
# Partially parsed test_hash_with_sha1_algorithm. Retrieved 3/6 statements.
# Partially parsed test_hash_with_md5_algorithm. Retrieved 3/6 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_maybe_probability_zero. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'value'
    var_2 = 0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_is_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_either_value_with_probability. Retrieved 8/12 statements.
# Partially parsed test_maybe_handles_non_string_values. Retrieved 8/12 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_out_of_range. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100
    var_5 = range(var_4)
    var_6 = 'other_value'
    var_7 = [closure(var_6, var_0) for _ in var_5]

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100
    var_5 = range(var_4)
    var_6 = 0
    var_7 = [closure(var_6, var_0) for _ in var_5]

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = -1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_different_input. Retrieved 11/15 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'sensitive_data'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'sensitive_data'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 12345
    var_3 = None
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_maybe_predicate_evaluates_to_false. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_suffix_adds_correct_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'ecipe'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #41
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_wrap_default_before_and_after. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_non_string_raises_type_error. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_only_before. Retrieved 3/4 statements.
# Partially parsed test_wrap_with_only_after. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = module_0.wrap(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ')'
    var_1 = module_0.wrap(after=var_0)
    var_2 = 'test'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_maybe_returns_value_with_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_first_argument_with_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_either_value_or_argument. Retrieved 6/9 statements.
# Partially parsed test_maybe_handles_zero_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_handles_one_probability. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = set()
    var_5 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 1
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other_value'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_join_returns_closure_that_joins_items_with_separator. Retrieved 7/8 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 2/4 statements.
# Partially parsed test_join_uses_default_comma_separator_when_none_provided. Retrieved 5/6 statements.
# Partially parsed test_join_handles_empty_iterable. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'pci | promise | excel'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = []



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_prefix_handles_empty_string_correctly. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_hash_with_valid_algorithm. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_maybe_with_probability_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_probability_negative. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 10

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 10



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_suffix_returns_correct_closure. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'recipe'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_prefix_raises_type_error_when_non_string_is_passed. Retrieved 5/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = None
    var_3 = 123
    var_4 = str(var_2)
    assert var_4 == 'prefix() requires a string, got int'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_apply_if_condition_false. Retrieved 8/9 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5



# Parsed testcases at query #52
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #54
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_join_predicate_evaluates_to_true. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_romanize_with_valid_locales. Retrieved 3/12 statements.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Привіт'
    var_2 = 'Сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_wrap_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_wrap_with_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_with_empty_string. Retrieved 2/3 statements.
# Partially parsed test_wrap_raises_type_error_for_non_string_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_wrap_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_wrap_with_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_with_empty_string. Retrieved 2/3 statements.
# Partially parsed test_wrap_raises_type_error_for_non_string_input. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_multiple_characters. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '<<<'
    var_1 = '>>>'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 3/7 statements.
# Partially parsed test_apply_if_condition_false_with_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_condition_false_without_otherwise. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_example_case. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'test'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'fields'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_false. Retrieved 5/10 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = '<'
    var_3 = '>'
    var_4 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_apply_if_with_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_condition_false. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_condition_false_and_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_and_otherwise. Retrieved 3/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 3
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = -5

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'test'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_truncate_no_truncation_needed. Retrieved 3/4 statements.
# Partially parsed test_truncate_truncation_needed. Retrieved 3/4 statements.
# Partially parsed test_truncate_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_max_length_equal_to_string_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_max_length_smaller_than_suffix_length. Retrieved 4/5 statements.
# Partially parsed test_truncate_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'this is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'this is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_suffix_adds_suffix_correctly. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_wrap_returns_correct_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_raises_type_error_for_non_string_input. Retrieved 4/6 statements.
# Partially parsed test_wrap_with_default_values. Retrieved 2/3 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'default'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_apply_if_condition_false. Retrieved 8/9 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5



# Parsed testcases at query #68
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_prefix_predicate_evaluates_to_false. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_iterable_raises_type_error. Retrieved 2/4 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_hash_with_supported_algorithm. Retrieved 3/5 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_apply_if_transform_applied_when_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_no_transform_when_condition_false. Retrieved 6/7 statements.
# Partially parsed test_apply_if_otherwise_transform_when_condition_false. Retrieved 8/9 statements.
# Partially parsed test_apply_if_no_transform_or_otherwise_when_condition_false. Retrieved 6/7 statements.
# Partially parsed test_apply_if_transform_applied_to_string. Retrieved 3/6 statements.
# Partially parsed test_apply_if_no_transform_to_short_string. Retrieved 3/6 statements.
# Partially parsed test_apply_if_otherwise_transform_to_short_string. Retrieved 3/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 3

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'word'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'cat'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'cat'



# Parsed testcases at query #74
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_truncate_predicate_evaluates_to_true. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a long string'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_apply_if_condition_false. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_suffix_predicate_false. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_wrap_default_values. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_values. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_non_string_input. Retrieved 4/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '{'
    var_1 = '}'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 123



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_iterable_raises_type_error. Retrieved 2/4 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_hash_with_supported_algorithm. Retrieved 3/5 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_suffix_adds_correct_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_suffix_handles_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_handles_non_alpha_numeric_characters. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = '!@#'



# Parsed testcases at query #84
#--------------------------

# Failed to parse test_romanize_with_valid_locale.
# Failed to parse test_romanize_with_unsupported_locale.
# Partially parsed test_romanize_returns_function_that_translates_string. Retrieved 1/6 statements.
# Partially parsed test_romanize_raises_type_error_for_non_string_input. Retrieved 1/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_prefix_with_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_with_non_string. Retrieved 3/5 statements.
# Partially parsed test_prefix_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_with_empty_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_with_both_empty. Retrieved 2/3 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/5 statements.
# Partially parsed test_hash_with_sha1_algorithm. Retrieved 3/6 statements.
# Partially parsed test_hash_with_md5_algorithm. Retrieved 3/6 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 2/4 statements.
# Partially parsed test_hash_with_empty_string. Retrieved 2/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = ''



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_truncate_returns_closure_that_truncates_string. Retrieved 4/5 statements.
# Partially parsed test_truncate_returns_original_string_when_shorter_than_max_length. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 4/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'abcdef'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 123



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_suffix_adds_suffix_correctly. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_join_raises_type_error_when_non_iterable_is_passed. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_condition_true_with_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_condition_false_with_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_condition_false_without_otherwise. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = -5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5



# Parsed testcases at query #91
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_true. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #93
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locales.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locales.
# Partially parsed test_romanize_closure_raises_type_error_for_non_string_input. Retrieved 1/7 statements.
# Partially parsed test_romanize_closure_translates_string_correctly. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_truncate_returns_closure_that_truncates_string. Retrieved 4/5 statements.
# Partially parsed test_truncate_returns_original_string_when_shorter_than_max_length. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 4/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'abcdefg'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 123



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_join_key_raises_type_error_for_non_iterable. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 123



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_maybe_predicate_evaluates_to_false. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'value'
    var_2 = 1.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'fallback'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_partial_functions. Retrieved 2/8 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.pipe()
    var_2 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_apply_if_transform_applied_when_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_otherwise_applied_when_condition_false. Retrieved 8/9 statements.
# Partially parsed test_apply_if_return_original_value_when_condition_false_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_string_condition. Retrieved 7/9 statements.
# Partially parsed test_apply_if_with_none_otherwise. Retrieved 6/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 6

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 4

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 4

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = 'hi'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 10



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_hash_with_sha256. Retrieved 3/4 statements.
# Partially parsed test_hash_with_md5. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_non_iterable_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_truncate_returns_full_string_when_within_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_adds_suffix_when_exceeds_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_uses_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_truncate_handles_empty_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_handles_string_equal_to_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_handles_unicode_strings. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello world'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello world'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'こんにちは世界'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_number. Retrieved 3/4 statements.
# Partially parsed test_redact_with_list. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'secret'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'secret'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locales.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locales.
# Partially parsed test_romanize_closure_raises_type_error_for_non_string_input. Retrieved 1/5 statements.
# Partially parsed test_romanize_closure_translates_string_correctly. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_wrap_default_behavior. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_before_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_string. Retrieved 2/3 statements.
# Partially parsed test_wrap_non_string_raises_typeerror. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_special_characters. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '{{'
    var_1 = '}}'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_truncate_with_long_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_short_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_exact_length_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Exactly 10'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '>>'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'This is a long string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_suffix_adds_suffix_correctly. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'ecipe'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_first_argument_with_complementary_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_either_value_or_first_argument_based_on_probability. Retrieved 6/9 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_is_out_of_range. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_is_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_value_when_probability_is_one. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'unexpected'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'unexpected'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = set()
    var_5 = 'unexpected'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = -1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'unexpected'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'unexpected'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'expected'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'unexpected'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_join_key_returns_joined_string. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = ' | '
    var_5 = module_0.join(var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pipe_with_multiple_functions. Retrieved 6/8 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 3/5 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/3 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 8/13 statements.


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = '!'
    var_2 = lambda x: x + var_1
    var_3 = 2
    var_4 = lambda x: x * var_3
    var_5 = 'test'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 1
    var_2 = 10
    var_3 = lambda x, random: x + str(random.randint(var_1, var_2))
    var_4 = module_0.Random()
    var_5 = 'test'
    var_6 = 'TEST'
    var_7 = -1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_wrap_raises_type_error_for_non_string_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_result_with_given_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_raises_no_error_with_probability_between_0_and_1. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_result_when_probability_is_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_returns_value_when_probability_is_one. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100
    var_5 = range(var_4)
    var_6 = [key_func(var_4, var_0) for _ in var_5]

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0.7
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pipe_evaluates_to_true. Retrieved 5/7 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = lambda x: x + var_1
    var_3 = 2
    var_4 = lambda x: x * var_3



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 4/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = None
    var_2 = 123
    var_3 = str(var_1)
    assert var_3 == 'join() requires iterable, got int'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_prefix_works_with_empty_string. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_hash_with_sha256. Retrieved 3/4 statements.
# Partially parsed test_hash_with_md5. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_wrap_raises_type_error_when_non_string_passed. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = None
    var_2 = 123



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_prefix_returns_correct_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_suffix_returns_closure_that_adds_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_suffix_works_with_empty_string. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'recip'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = ''



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 7/8 statements.
# Partially parsed test_apply_if_condition_false_with_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_condition_false_without_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_strings. Retrieved 7/9 statements.
# Partially parsed test_apply_if_with_none_otherwise. Retrieved 6/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'word'
    var_6 = 'the'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_join_closure_joins_list_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_closure_joins_list_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_closure_converts_items_to_string. Retrieved 5/6 statements.
# Partially parsed test_join_closure_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = callable(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #28
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_prefix_returns_closure_that_adds_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_prefix_works_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_works_with_empty_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_works_with_multiple_calls. Retrieved 4/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'name'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_suffix_predicate_evaluates_to_false. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'



# Parsed testcases at query #31
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_number_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_list_input. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'secret'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_apply_if_condition_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 'test'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_truncate_no_truncation_needed. Retrieved 3/4 statements.
# Partially parsed test_truncate_truncation_with_default_suffix. Retrieved 3/4 statements.
# Partially parsed test_truncate_truncation_with_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_exact_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_edge_case_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_raises_type_error. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'this is a long sentence'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'this is a long sentence'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'exactly10'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'justover10'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_integer_input. Retrieved 3/4 statements.
# Partially parsed test_redact_with_list_input. Retrieved 6/7 statements.
# Partially parsed test_redact_with_dict_input. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'sensitive_data'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'sensitive_data'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_non_iterable_raises_type_error. Retrieved 2/4 statements.
# Partially parsed test_join_with_custom_iterable. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 'cherry'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '; '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_suffix_returns_closure_that_adds_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = False
    var_3 = 123
    var_4 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pipe_functions. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_none_value. Retrieved 3/4 statements.
# Partially parsed test_redact_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_redact_with_int_value. Retrieved 3/4 statements.
# Partially parsed test_redact_with_list_value. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[REDACTED]'
    var_1 = module_0.redact(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_apply_if_condition_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 'test'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_apply_if_predicate_false. Retrieved 8/9 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_join_raises_type_error_for_non_iterable_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 42



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_truncate_predicate_evaluates_to_true. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a long string'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_prefix_predicate_evaluates_to_false. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_wrap_default_before_and_after. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_string. Retrieved 4/5 statements.
# Partially parsed test_wrap_non_string_raises_typeerror. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_special_characters. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '{'
    var_1 = '}'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'a&b'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_romanize_with_valid_locale_ru. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_valid_locale_uk. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_valid_locale_kk. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'kk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_false. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_suffix_adds_suffix_correctly. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_suffix_works_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_works_with_empty_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_works_with_multiple_char_suffix. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '_suffix'
    var_1 = module_0.suffix(var_0)
    var_2 = 'text'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_true. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_validate_locale_with_unsupported_locale.




# Parsed testcases at query #55
#--------------------------

# Partially parsed test_truncate_returns_original_string_when_shorter_than_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_returns_truncated_string_when_longer_than_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_uses_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'longer'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'longer'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 5/8 statements.
# Partially parsed test_truncate_returns_original_string_when_shorter_than_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_returns_truncated_string_when_longer_than_max_length. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'test'
    var_3 = module_0.truncate(var_0)
    var_4 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'longer'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_join_with_non_iterable_input_raises_type_error. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 123



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_hash_with_supported_algorithm. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_element. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_elements. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = [var_1]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_hash_with_sha256. Retrieved 3/4 statements.
# Partially parsed test_hash_with_md5. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_true. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_wrap_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_wrap_with_custom_before_and_after. Retrieved 4/5 statements.
# Partially parsed test_wrap_with_empty_string. Retrieved 2/3 statements.
# Partially parsed test_wrap_with_non_string_input_raises_typeerror. Retrieved 2/4 statements.
# Partially parsed test_wrap_with_multiple_characters. Retrieved 4/5 statements.
# Partially parsed test_wrap_with_special_characters. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '<<<'
    var_1 = '>>>'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '$'
    var_1 = '%'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_hash_with_sha256. Retrieved 3/4 statements.
# Partially parsed test_hash_with_md5. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123



# Parsed testcases at query #67
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_suffix_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_suffix_adds_correct_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'ecipe'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 3/7 statements.
# Partially parsed test_apply_if_condition_true_without_otherwise. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_apply_if_transform_applied_when_condition_true. Retrieved 6/7 statements.
# Partially parsed test_apply_if_otherwise_applied_when_condition_false. Retrieved 8/9 statements.
# Partially parsed test_apply_if_no_transform_applied_when_condition_false_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_transform_applied_to_string. Retrieved 5/6 statements.
# Partially parsed test_apply_if_otherwise_applied_to_string. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 3
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = -5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = lambda x: x.upper()
    var_3 = module_0.apply_if(var_1, var_2)
    var_4 = 'word'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'hi'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/5 statements.
# Partially parsed test_hash_with_sha1_algorithm. Retrieved 3/6 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_join_returns_correct_closure. Retrieved 6/7 statements.
# Partially parsed test_join_raises_type_error_for_non_iterable. Retrieved 2/4 statements.
# Partially parsed test_join_handles_empty_list. Retrieved 3/4 statements.
# Partially parsed test_join_handles_list_with_one_item. Retrieved 4/5 statements.
# Partially parsed test_join_handles_list_with_multiple_types. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 'single'
    var_3 = [var_2]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 'two'
    var_4 = 3.0
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_join_returns_closure_that_joins_items_with_separator. Retrieved 6/7 statements.
# Partially parsed test_join_uses_default_comma_separator_when_none_provided. Retrieved 5/6 statements.
# Partially parsed test_join_raises_type_error_when_non_iterable_passed. Retrieved 2/4 statements.
# Partially parsed test_join_handles_empty_iterable. Retrieved 3/4 statements.
# Partially parsed test_join_converts_non_string_items_to_string. Retrieved 6/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = ':'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2.5
    var_4 = True
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_romanize_raises_value_error_for_unsupported_locale. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'en'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_prefix_returns_closure_that_adds_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_prefix_works_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_works_with_empty_prefix. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #78
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_prefix_predicate_evaluates_to_false. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



