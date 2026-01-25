####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_not_met. Retrieved 7/8 statements.
# Partially parsed test_maybe_returns_value_when_probability_one. Retrieved 6/7 statements.
# Partially parsed test_maybe_works_with_different_value_types. Retrieved 7/8 statements.
# Partially parsed test_maybe_works_with_none_value. Retrieved 7/8 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = 0.7
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'special'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 'special'
    var_3 = 0.3
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = module_1.maybe(var_2, var_1)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'special'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 123
    var_3 = 0.8
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 456
    var_6 = var_4(var_5, var_0)
    assert var_6 == 123

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = None
    var_3 = 0.6
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'not_none'
    var_6 = var_4(var_5, var_0)
    assert var_6 is None

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 1.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'



# Parsed testcases at query #2
#--------------------------




import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'
    var_5 = var_3(var_4, var_0)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'
    var_5 = var_3(var_4, var_0)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 1.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'
    var_5 = var_3(var_4, var_0)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_without_otherwise. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_with_otherwise. Retrieved 3/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_8 = var_6(var_7)
    assert var_8 == -15

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'HI'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 10
    var_4 = lambda x: x * var_3
    var_5 = None
    var_6 = module_0.apply_if(var_2, var_4, var_5)
    var_7 = 3
    var_8 = var_6(var_7)
    assert var_8 == 3

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = lambda d: var_0 in d
    var_2 = 'processed'
    var_3 = True
    var_4 = lambda d: {var_2: d}
    var_5 = module_0.apply_if(var_1, var_4)
    var_6 = 'value'
    var_7 = 10
    var_8 = {var_0: var_3, var_6: var_7}
    var_9 = var_5(var_8)
    var_10 = bool(var_9 == {'valid': True, 'value': 10, 'processed': True})
    assert var_10 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = lambda d: var_0 in d
    var_2 = 'processed'
    var_3 = True
    var_4 = lambda d: {var_2: d}
    var_5 = False
    var_6 = lambda d: {var_2: d}
    var_7 = module_0.apply_if(var_1, var_4, var_6)
    var_8 = 'value'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = var_7(var_10)
    var_12 = bool(var_11 == {'value': 10, 'processed': False})
    assert var_12 is True



# Parsed testcases at query #4
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = var_1(var_2)
    assert var_3 == 'user_order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    assert var_3 == 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = None
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'pre_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'fix@123'
    var_3 = var_1(var_2)
    assert var_3 == 'pre_fix@123'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'pre_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == 'pre_'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'one'
    var_3 = var_1(var_2)
    assert var_3 == 'test_one'
    var_4 = 'two'
    var_5 = var_1(var_4)
    assert var_5 == 'test_two'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_romanize_returns_callable_for_supported_locales. Retrieved 4/18 statements.
# Partially parsed test_romanize_raises_value_error_for_unsupported_locale. Retrieved 2/6 statements.
# Partially parsed test_romanize_accepts_locale_string. Retrieved 4/7 statements.
# Partially parsed test_returned_function_romanizes_russian_text. Retrieved 1/6 statements.
# Partially parsed test_returned_function_romanizes_ukrainian_text. Retrieved 1/6 statements.
# Partially parsed test_returned_function_romanizes_kazakh_text. Retrieved 1/6 statements.
# Partially parsed test_returned_function_handles_common_letters. Retrieved 1/6 statements.
# Partially parsed test_returned_function_raises_type_error_for_non_string_input. Retrieved 3/8 statements.
# Partially parsed test_returned_function_returns_empty_string_for_empty_input. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = [var_1]
    var_3 = lambda : var_0
    var_4 = [var_3]
    var_5 = lambda : var_0
    var_6 = [var_5]

def test_case_0():
    var_0 = False
    var_1 = True
    assert var_1 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = None
    var_3 = lambda : var_2
    var_4 = [var_3]

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True

def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 'Привіт'

def test_case_0():
    var_0 = 'Сәлем'

def test_case_0():
    var_0 = 'ёж'

def test_case_0():
    var_0 = 123
    var_1 = False
    var_2 = True
    assert var_2 is True

def test_case_0():
    var_0 = ''



# Parsed testcases at query #6
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'
    var_3 = var_1(var_2)
    assert var_3 == 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'exact'
    var_3 = var_1(var_2)
    assert var_3 == 'exact'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'this is a long string'
    var_3 = var_1(var_2)
    assert var_3 == 'this is...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'this is a long string'
    var_4 = var_2(var_3)
    assert var_4 == 'this is !!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = -5
    var_1 = module_0.truncate(var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == '..'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == '...'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #8
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Ports are created'
    var_3 = var_1(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 <= 20)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_romanize_raises_error_for_unsupported_locale.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pipe_with_key_functions_using_random. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_key_functions_without_random. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_mixed_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_empty_functions. Retrieved 1/4 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/6 statements.
# Partially parsed test_pipe_with_nested_pipe. Retrieved 1/9 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = '  HELLO  '

def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = 3



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locales.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanizer_raises_type_error_for_non_string_input. Retrieved 1/7 statements.
# Partially parsed test_romanizer_translates_russian_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_translates_ukrainian_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_translates_kazakh_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_handles_empty_string. Retrieved 1/6 statements.
# Partially parsed test_romanizer_handles_common_letters. Retrieved 2/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'Привет мир'
    var_1 = 'Privet mir'

def test_case_0():
    var_0 = 'Привіт світ'
    var_1 = 'Pryvit svit'

def test_case_0():
    var_0 = 'Сәлем әлем'
    var_1 = 'Sälem älem'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'ёж'
    var_1 = 'yozh'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_validate_locale_returns_locale_when_locale_is_locale_instance.




# Parsed testcases at query #13
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #14
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locales.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanize_raises_type_error_for_non_string_input. Retrieved 1/7 statements.
# Partially parsed test_romanize_raises_type_error_for_none_input. Retrieved 1/7 statements.
# Partially parsed test_romanize_translates_russian_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_translates_ukrainian_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_translates_kazakh_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_handles_empty_string. Retrieved 1/6 statements.
# Partially parsed test_romanize_handles_mixed_case. Retrieved 1/6 statements.
# Partially parsed test_romanize_handles_common_letters. Retrieved 1/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = None

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 'Привіт'

def test_case_0():
    var_0 = 'Сәлем'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'ПрИвЕт'

def test_case_0():
    var_0 = 'ёЁ'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pipe_handles_key_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_handles_key_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 3/14 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 3/12 statements.
# Partially parsed test_pipe_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_argument. Retrieved 3/8 statements.
# Partially parsed test_pipe_chain_with_and_without_random. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_'
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = '!'
    var_1 = 'hi'
    var_2 = None

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'anything'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 7
    var_2 = 'num'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'b'
    var_2 = 'test'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #21
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #22
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'John Doe'
    var_1 = None



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #28
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #29
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pipe_with_random_parameter. Retrieved 2/8 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_validate_locale_returns_locale_when_locale_is_locale_instance.




# Parsed testcases at query #33
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'recipe'
    var_3 = var_1(var_2)
    assert var_3 == 'recipe.io'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 == 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '_suffix'
    var_1 = module_0.suffix(var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    assert var_3 == 'test_suffix'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = None
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.suffix(var_0)
    var_2 = 'a'
    var_3 = var_1(var_2)
    assert var_3 == 'a.'
    var_4 = 'b'
    var_5 = var_1(var_4)
    assert var_5 == 'b.'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '!@#$'
    var_1 = module_0.suffix(var_0)
    var_2 = 'word'
    var_3 = var_1(var_2)
    assert var_3 == 'word!@#$'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '!'
    var_1 = module_0.suffix(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == '!'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_maybe_returns_value_with_given_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_returns_first_argument_when_probability_not_met. Retrieved 7/8 statements.
# Partially parsed test_maybe_works_with_different_value_types. Retrieved 7/8 statements.
# Partially parsed test_maybe_works_with_none_value. Retrieved 7/8 statements.
# Partially parsed test_maybe_uses_correct_weights. Retrieved 8/13 statements.


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = 0.7
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'special'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 'special'
    var_3 = 0.3
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 1.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'special'

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 42
    var_3 = 0.8
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 0
    var_6 = var_4(var_5, var_0)
    assert var_6 == 42

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = None
    var_3 = 0.6
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'not_none'
    var_6 = var_4(var_5, var_0)
    assert var_6 is None

import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = []
    var_2 = []
    var_3 = 'value'
    var_4 = 0.75
    var_5 = module_1.maybe(var_3, var_4)
    var_6 = 'result'
    var_7 = var_5(var_6, var_0)
    var_8 = bool(var_1 == ['result', 'value'])
    assert var_8 is True
    var_9 = bool(var_2 == [0.25, 0.75])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/11 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/8 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 2

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'hello'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = []
    var_2 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5



# Parsed testcases at query #4
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = var_1(var_2)
    assert var_3 == 'user_order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = var_1(var_2)
    assert var_3 == 'order'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == 'user_'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'pre_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'test@123'
    var_3 = var_1(var_2)
    assert var_3 == 'pre_test@123'



# Parsed testcases at query #5
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(var_4)
    assert var_5 == 'a, b, c'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1(var_5)
    assert var_6 == 'pci | promise | excel'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = []
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = ' '
    var_1 = module_0.join(var_0)
    var_2 = 'hello'
    var_3 = [var_2]
    var_4 = var_1(var_3)
    assert var_4 == 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2.5
    var_4 = True
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1(var_5)
    assert var_6 == '1, 2.5, True'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 123
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_without_otherwise. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_with_otherwise. Retrieved 3/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_8 = var_6(var_7)
    assert var_8 == -15

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
    var_8 = var_6(var_7)
    assert var_8 == 10

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'HI'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'transformed'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 == 'transformed'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'transformed'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 'not_none'
    var_6 = var_4(var_5)
    assert var_6 == 'not_none'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = lambda x: x * var_0
    var_3 = None
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 5



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_romanize_with_valid_locale_ru. Retrieved 1/4 statements.
# Partially parsed test_romanize_with_valid_locale_uk. Retrieved 1/4 statements.
# Partially parsed test_romanize_with_valid_locale_kk. Retrieved 1/4 statements.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanize_raises_type_error_for_non_string_input. Retrieved 1/5 statements.
# Partially parsed test_romanize_common_letters_translation. Retrieved 1/4 statements.
# Partially parsed test_romanize_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 'Привіт'

def test_case_0():
    var_0 = 'Сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'
    var_3 = var_1(var_2)
    assert var_3 == 'Privet'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 'ёж'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #8
#--------------------------




import mimesis.locales as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.validate_locale(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 4/16 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/8 statements.
# Partially parsed test_pipe_with_nested_pipes. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'pre-'
    var_1 = '-suf'
    var_2 = 'middle'
    var_3 = None

def test_case_0():
    var_0 = 'ab'
    var_1 = None

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 7
    var_2 = 'value'

def test_case_0():
    var_0 = 3
    var_1 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_2_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 3
    var_2 = lambda x: len(x) > var_1
    var_3 = var_2(var_0)
    assert var_3 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pipe_applies_functions_in_sequence. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 2/9 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/9 statements.
# Partially parsed test_pipe_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/7 statements.
# Partially parsed test_pipe_handles_type_error. Retrieved 3/10 statements.
# Partially parsed test_pipe_chain_of_three_functions. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = []
    var_1 = 42

def test_case_0():
    var_0 = '  HELLO  '

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3
    var_2 = 5

def test_case_0():
    var_0 = 2



# Parsed testcases at query #12
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 5

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = lambda x: x.startswith(var_0)
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'Banana'
    var_6 = var_4(var_5)
    assert var_6 == 'banana'

import mimesis.keys as module_0

def test_case_0():
    var_0 = lambda x: bool(x)
    var_1 = 'truthy'
    var_2 = lambda x: var_1
    var_3 = 'falsy'
    var_4 = lambda x: var_3
    var_5 = module_0.apply_if(var_0, var_2, var_4)
    var_6 = 0
    var_7 = var_5(var_6)
    assert var_7 == 'falsy'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 'not none'
    var_3 = lambda x: var_2
    var_4 = 'none'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 'none'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = 'non-empty'
    var_3 = lambda x: var_2
    var_4 = 'empty'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = ''
    var_8 = var_6(var_7)
    assert var_8 == 'empty'



# Parsed testcases at query #13
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'
    var_3 = var_1(var_2)
    assert var_3 == 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'exact'
    var_3 = var_1(var_2)
    assert var_3 == 'exact'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a long sentence.'
    var_3 = var_1(var_2)
    assert var_3 == 'This i...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'This is a long sentence.'
    var_4 = var_2(var_3)
    assert var_4 == 'This is !!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == '...'



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locale.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanizer_raises_type_error_for_non_string_input. Retrieved 1/7 statements.
# Partially parsed test_romanizer_translates_russian_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_translates_ukrainian_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_translates_kazakh_text. Retrieved 2/7 statements.
# Partially parsed test_romanizer_handles_empty_string. Retrieved 1/6 statements.
# Partially parsed test_romanizer_handles_common_letters. Retrieved 2/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Privet'

def test_case_0():
    var_0 = 'Привіт'
    var_1 = 'Pryvit'

def test_case_0():
    var_0 = 'Сәлем'
    var_1 = 'Sälem'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'ёЁ'
    var_1 = 'eË'



# Parsed testcases at query #15
#--------------------------

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
    var_6 = var_4(var_5)
    assert var_6 == 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_8 = var_6(var_7)
    assert var_8 == -15

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
    var_8 = var_6(var_7)
    assert var_8 == 10

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'HI'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 'not none'
    var_3 = lambda x: var_2
    var_4 = 'none'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 'none'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda lst: len(lst) > var_0
    var_2 = 4
    var_3 = [var_2]
    var_4 = lambda lst: lst + var_3
    var_5 = module_0.apply_if(var_1, var_4)
    var_6 = 1
    var_7 = 3
    var_8 = [var_6, var_0, var_7]
    var_9 = var_5(var_8)
    var_10 = bool(var_9 == [1, 2, 3, 4])
    assert var_10 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda lst: len(lst) > var_0
    var_2 = 4
    var_3 = [var_2]
    var_4 = lambda lst: lst + var_3
    var_5 = module_0.apply_if(var_1, var_4)
    var_6 = 1
    var_7 = [var_6, var_0]
    var_8 = var_5(var_7)
    var_9 = bool(var_8 == [1, 2])
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_with_random_parameter. Retrieved 4/21 statements.
# Partially parsed test_pipe_without_random_parameter. Retrieved 2/8 statements.
# Partially parsed test_pipe_mixed_functions. Retrieved 3/12 statements.
# Partially parsed test_pipe_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_empty_functions. Retrieved 2/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 'TEST_'
    var_1 = 'hello'
    var_2 = module_0.Random()
    var_3 = 5

def test_case_0():
    var_0 = 5
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = '!'
    var_1 = 'hello'
    var_2 = module_0.Random()

def test_case_0():
    var_0 = 4
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = module_0.Random()



# Parsed testcases at query #17
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 5
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_4)
    assert var_7 == 10



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #19
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x / var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    var_8 = bool(var_7 == 2.5)
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 6



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pipe_applies_functions_in_sequence. Retrieved 1/7 statements.
# Partially parsed test_pipe_handles_key_func_with_random. Retrieved 5/14 statements.
# Partially parsed test_pipe_handles_mixed_functions. Retrieved 5/13 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_preserves_type_handling. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = 2

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = 10

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = []
    var_1 = 42

def test_case_0():
    var_0 = 100



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pipe_key_function_with_random_parameter. Retrieved 4/22 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'TEST-'
    var_2 = 'hello'
    var_3 = 5



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_regular_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_chain_of_three. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 4
    var_1 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3
    var_2 = 5

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = None

def test_case_0():
    var_0 = 3
    var_1 = None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pipe_with_key_functions_using_random. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_key_functions_without_random. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_mixed_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/6 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = '  HELLO  '

def test_case_0():
    var_0 = []
    var_1 = 42



