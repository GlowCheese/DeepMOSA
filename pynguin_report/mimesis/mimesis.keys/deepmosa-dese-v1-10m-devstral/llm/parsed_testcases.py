####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.
# Partially parsed test_romanize_with_empty_string. Retrieved 1/4 statements.
# Partially parsed test_romanize_with_mixed_case. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'ПриВет'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_suffix_adds_correct_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_with_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'recipe'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 4/6 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string. Retrieved 5/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'
    var_3 = 'id'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123
    var_3 = 'TypeError not raised'
    var_4 = AssertionError(var_3)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_'
    var_1 = module_0.prefix(var_0)
    var_2 = callable(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
    var_2 = [var_1]

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
    var_1 = 'not iterable'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input_value. Retrieved 7/10 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'REDACTED'
    var_1 = module_0.redact(var_0)
    var_2 = None
    var_3 = 123
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_apply_if_with_true_condition. Retrieved 7/8 statements.
# Partially parsed test_apply_if_with_false_condition_and_otherwise. Retrieved 7/8 statements.
# Partially parsed test_apply_if_with_false_condition_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_none_otherwise. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = -5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_maybe_with_valid_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_probability_zero. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_probability_one. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_invalid_probability_negative. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_invalid_probability_above_one. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.7
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input. Retrieved 6/10 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '***'
    var_1 = module_0.redact(var_0)
    var_2 = 'input1'
    var_3 = 'input2'
    var_4 = None
    var_5 = 123



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_wrap_default. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom. Retrieved 4/5 statements.
# Partially parsed test_wrap_type_error. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/3 statements.
# Partially parsed test_hash_with_sha1. Retrieved 3/4 statements.
# Partially parsed test_hash_with_md5. Retrieved 3/4 statements.
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
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0.6
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'hello'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'HELLO'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 4/5 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 'hello'
    var_3 = [var_2]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ';'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2.5
    var_4 = True
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'not_iterable'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_suffix_adds_correctly. Retrieved 4/6 statements.
# Partially parsed test_suffix_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'word'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_suffix_predicate_false. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'Add suffix to result.'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'only'
    var_2 = [var_1]

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
    var_1 = 'not_iterable'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe'



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input_value. Retrieved 5/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'REDACTED'
    var_1 = module_0.redact(var_0)
    var_2 = 'input1'
    var_3 = 'input2'
    var_4 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_prefix_closure_adds_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_closure_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = callable(var_1)

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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
    var_2 = [var_1]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'not iterable'



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_prefix_predicate. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #24
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = callable(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_wrap_default. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_strings. Retrieved 3/4 statements.
# Partially parsed test_wrap_non_string_input. Retrieved 2/4 statements.


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
    var_0 = ''
    var_1 = module_0.wrap(var_0, var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_wrap_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Wrap result with before and after strings.'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_apply_if_with_true_condition. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_false_condition_and_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_false_condition_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_and_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_none_condition. Retrieved 7/8 statements.
# Partially parsed test_apply_if_with_none_condition_and_otherwise. Retrieved 8/9 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 3
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5

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

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'is None'
    var_3 = lambda x: var_2
    var_4 = 'not None'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'is None'
    var_3 = lambda x: var_2
    var_4 = 'not None'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 'something'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input. Retrieved 6/10 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'REDACTED'
    var_1 = module_0.redact(var_0)
    var_2 = 'input1'
    var_3 = 'input2'
    var_4 = None
    var_5 = 123



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_apply_if_with_true_condition. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_false_condition_and_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_false_condition_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_string_condition_and_transform. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_and_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_none_otherwise. Retrieved 5/6 statements.
# Partially parsed test_apply_if_with_none_value_and_otherwise. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 3
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5

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

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = 0
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/4 statements.
# Partially parsed test_hash_with_custom_algorithm. Retrieved 3/5 statements.
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
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_maybe_closure_accepts_two_args. Retrieved 4/6 statements.
# Partially parsed test_maybe_returns_value_with_high_probability. Retrieved 8/9 statements.
# Partially parsed test_maybe_returns_original_with_low_probability. Retrieved 8/9 statements.
# Partially parsed test_maybe_handles_zero_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_handles_one_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_works_with_different_types. Retrieved 4/6 statements.
# Partially parsed test_maybe_preserves_none_values. Retrieved 4/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe(var_0)
    var_2 = module_1.Random()
    var_3 = 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.99
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 1000
    var_5 = range(var_4)
    var_6 = 100
    var_7 = [key_func(var_6, var_3) for _ in var_5]

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.01
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 1000
    var_5 = range(var_4)
    var_6 = 100
    var_7 = [key_func(var_6, var_3) for _ in var_5]

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.maybe(var_0)
    var_2 = module_1.Random()
    var_3 = 'world'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe(var_0)
    var_2 = module_1.Random()
    var_3 = 42



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_truncate_with_max_length_less_than_string_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_equal_to_string_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_greater_than_string_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_exact_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_raises_type_error_for_non_string_input. Retrieved 3/5 statements.
# Partially parsed test_truncate_with_max_length_one. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_two. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_three. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_four. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_five. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_six. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_seven. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_eight. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_nine. Retrieved 3/4 statements.
# Partially parsed test_truncate_with_max_length_ten. Retrieved 3/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = '!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.truncate(var_0)
    var_2 = 'abc'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 6
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 7
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 8
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 9
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_join_predicate_false.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_join_predicate.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
    var_2 = [var_1]

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
    var_1 = 'not a list'



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0.6
    var_2 = 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'hello'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'hello'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 1/5 statements.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)

import builtins as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_1.romanize(var_0)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_truncate_basic. Retrieved 4/6 statements.
# Partially parsed test_truncate_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_no_truncation_needed. Retrieved 3/4 statements.
# Partially parsed test_truncate_exact_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_empty_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'
    var_3 = 'Short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Short string'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = -5
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #43
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported_algorithm'
    var_1 = module_0.hash_with(var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_maybe_with_valid_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_zero_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_one_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_invalid_negative_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_invalid_above_one_probability. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.7
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_join_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Join list items with separator.'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/3 statements.
# Partially parsed test_hash_with_sha1. Retrieved 3/4 statements.
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
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/12 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'
    var_1 = 'привіт'
    var_2 = 'кеш'

def test_case_0():
    var_0 = 123



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_romanize_raises_valueerror_for_unsupported_locale.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_truncate_within_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_exceeds_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_empty_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_exact_max_length. Retrieved 3/4 statements.
# Partially parsed test_truncate_raises_type_error. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 7
    var_1 = '~'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Testing'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.truncate(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 4/5 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 'hello'
    var_3 = [var_2]

import mimesis.keys as module_0

def test_case_0():
    var_0 = ';'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'not_iterable'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_with_different_input_types. Retrieved 11/15 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'string'
    var_2 = module_0.redact()
    var_3 = 123
    var_4 = module_0.redact()
    var_5 = None
    var_6 = module_0.redact()
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_apply_if_with_true_condition. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_false_condition_and_no_otherwise. Retrieved 6/7 statements.
# Partially parsed test_apply_if_with_false_condition_and_otherwise. Retrieved 8/9 statements.
# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.


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
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input_value. Retrieved 9/12 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'X'
    var_1 = module_0.redact(var_0)
    var_2 = None
    var_3 = 'Y'
    var_4 = module_0.redact(var_3)
    var_5 = 42
    var_6 = 'Z'
    var_7 = module_0.redact(var_6)
    var_8 = 'some_string'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_wrap_default_brackets. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom_brackets. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty_strings. Retrieved 3/4 statements.
# Partially parsed test_wrap_raises_type_error. Retrieved 2/4 statements.


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
    var_0 = ''
    var_1 = module_0.wrap(var_0, var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_join_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Join list items with separator.'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_maybe_closure_returns_value_or_result. Retrieved 5/6 statements.
# Partially parsed test_maybe_closure_returns_result_with_zero_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_closure_returns_result_with_invalid_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_closure_returns_result_with_high_probability. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 50

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 50

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 50

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = 2.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 50



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_wrap_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Wrap result with before and after strings.'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 3/4 statements.
# Partially parsed test_hash_with_custom_algorithm. Retrieved 3/4 statements.
# Partially parsed test_hash_with_non_string_input. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = callable(var_0)
    var_2 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'password'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/9 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 4/6 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 100
    var_3 = lambda x, r: x + str(r.randint(var_1, var_2))
    var_4 = 'value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'test'
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'test'
    var_3 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 100
    var_3 = lambda x, r: x + str(r.randint(var_1, var_2))
    var_4 = 'TEST'
    var_5 = 'test'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_maybe_closure_returns_original_value. Retrieved 5/6 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original_value'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
# Partially parsed test_join_with_non_string_items. Retrieved 6/7 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
    var_2 = [var_1]

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
    var_1 = 123



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 4/7 statements.
# Partially parsed test_pipe_handles_none_random. Retrieved 3/5 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/9 statements.
# Partially parsed test_pipe_empty_functions_list. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_type_error_handling. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = None

def test_case_0():
    var_0 = None
    var_1 = lambda x, r: x if r is var_0 else x.upper()
    var_2 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 9
    var_3 = lambda x, r: x + str(r.randint(var_1, var_2))
    var_4 = 'value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'unchanged'
    var_2 = None

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'test'
    var_2 = None



# Parsed testcases at query #21
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #22
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = callable(var_0)



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_truncate_basic. Retrieved 5/8 statements.
# Partially parsed test_truncate_custom_suffix. Retrieved 4/5 statements.
# Partially parsed test_truncate_empty_string. Retrieved 3/4 statements.
# Partially parsed test_truncate_exact_length. Retrieved 4/6 statements.
# Partially parsed test_truncate_non_string_input. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'
    var_3 = 'Short'
    var_4 = 'Exactly10'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'
    var_3 = 'Hello!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)
    var_2 = -5
    var_3 = module_0.truncate(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_suffix_adds_correct_suffix. Retrieved 3/4 statements.
# Partially parsed test_suffix_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_suffix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_wrap_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
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
    var_1 = 'not a list'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_ignores_input_value. Retrieved 6/9 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'password'
    var_2 = module_0.redact()
    var_3 = '12345'
    var_4 = module_0.redact()
    var_5 = ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_condition_evaluates_to_true. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x / var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_wrap_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Wrap result with before and after strings.'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_join_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
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
    var_1 = 'not a list'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_prefix_returns_closure_that_adds_prefix. Retrieved 4/6 statements.
# Partially parsed test_prefix_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'
    var_3 = 'age'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'pre_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
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
    var_1 = 'not_iterable'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/3 statements.
# Partially parsed test_hash_with_sha1. Retrieved 3/4 statements.
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
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_for_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_join_with_default_separator. Retrieved 5/6 statements.
# Partially parsed test_join_with_custom_separator. Retrieved 6/7 statements.
# Partially parsed test_join_with_empty_list. Retrieved 2/3 statements.
# Partially parsed test_join_with_single_item. Retrieved 3/4 statements.
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
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = []

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'single'
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
    var_1 = 'not iterable'



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_redact_default_replacement. Retrieved 2/3 statements.
# Partially parsed test_redact_custom_replacement. Retrieved 3/4 statements.
# Partially parsed test_redact_different_input_types. Retrieved 7/10 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'REDACTED'
    var_1 = module_0.redact(var_0)
    var_2 = 123
    var_3 = None
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_hash_with_default_algorithm. Retrieved 2/3 statements.
# Partially parsed test_hash_with_sha1. Retrieved 3/4 statements.
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
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_wrap_predicate_false. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = '<test>'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pipe_docstring_is_correct. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_maybe_closure_returns_value_with_probability. Retrieved 6/8 statements.
# Partially parsed test_maybe_closure_returns_result_with_probability. Retrieved 6/8 statements.
# Partially parsed test_maybe_closure_returns_result_when_probability_out_of_range. Retrieved 5/6 statements.
# Partially parsed test_maybe_closure_returns_result_when_probability_above_range. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = [var_1]
    var_3 = 1.0
    var_4 = module_1.maybe(var_1, var_3)
    var_5 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = [var_1]
    var_3 = 42
    var_4 = 0.0
    var_5 = module_1.maybe(var_3, var_4)

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 100



# Parsed testcases at query #46
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_maybe_with_default_probability. Retrieved 4/5 statements.
# Partially parsed test_maybe_with_custom_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_zero_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_one_probability. Retrieved 5/6 statements.
# Partially parsed test_maybe_with_invalid_probability. Retrieved 5/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = module_1.Random()
    var_3 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.8
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_romanize_with_russian_locale. Retrieved 2/6 statements.
# Partially parsed test_romanize_with_ukrainian_locale. Retrieved 2/6 statements.
# Partially parsed test_romanize_with_kazakh_locale. Retrieved 2/6 statements.
# Failed to parse test_romanize_with_unsupported_locale.
# Partially parsed test_romanize_with_invalid_string_input. Retrieved 1/5 statements.
# Partially parsed test_romanize_with_string_locale. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Москва'

def test_case_0():
    var_0 = 'Привіт'
    var_1 = 'Київ'

def test_case_0():
    var_0 = 'Сәлем'
    var_1 = 'Алматы'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_handles_type_error_without_random. Retrieved 4/6 statements.
# Partially parsed test_pipe_handles_type_error_with_random. Retrieved 5/8 statements.
# Partially parsed test_pipe_returns_unchanged_value_with_no_functions. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'HELLO'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = lambda x, r: x + r.randint(var_1, var_2)
    var_4 = 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'test'
    var_2 = None



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_wrap_default. Retrieved 2/3 statements.
# Partially parsed test_wrap_custom. Retrieved 4/5 statements.
# Partially parsed test_wrap_empty. Retrieved 3/4 statements.
# Partially parsed test_wrap_non_string. Retrieved 2/4 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.wrap(var_0, var_0)
    var_2 = 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_prefix_adds_correct_prefix. Retrieved 3/4 statements.
# Partially parsed test_prefix_raises_type_error_on_non_string. Retrieved 3/5 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_join_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/9 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 3/4 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 100
    var_3 = lambda x, r: x + str(r.randint(var_1, var_2))
    var_4 = 'value'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.pipe()
    var_1 = 'test'
    var_2 = None

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'test'
    var_2 = None



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_suffix_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Add suffix to result.'



# Parsed testcases at query #55
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = callable(var_1)



