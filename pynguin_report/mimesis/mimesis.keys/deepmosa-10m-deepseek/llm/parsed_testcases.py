####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_romanize_returns_callable_for_supported_locale.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanize_closure_romanizes_russian_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_closure_romanizes_ukrainian_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_closure_romanizes_kazakh_text. Retrieved 1/6 statements.
# Partially parsed test_romanize_closure_raises_type_error_for_non_string_input. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 'Привіт'

def test_case_0():
    var_0 = 'Сәлем'

def test_case_0():
    var_0 = 123
    var_1 = 'romanize() requires a string, got'

import mimesis.keys as module_0


def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'
    var_3 = var_1(var_2)
    assert var_3 == 'Privet'


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)
    var_2 = 'invalid'


def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)
    var_2 = '123'



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = var_0(var_1)
    assert var_2 == '<test>'


def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'
    var_4 = var_2(var_3)
    assert var_4 == '[test]'


def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = ''
    var_4 = var_2(var_3)
    assert var_4 == '()'


def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.wrap()
    var_1 = None
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = var_0(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '***'
    var_1 = module_0.wrap(var_0, var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    assert var_3 == '***test***'


def test_case_0():
    var_0 = '{'
    var_1 = '}'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'first'
    var_4 = var_2(var_3)
    assert var_4 == '{first}'
    var_5 = 'second'
    var_6 = var_2(var_5)
    assert var_6 == '{second}'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 3/12 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_multiple_functions_no_random. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 3/12 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/8 statements.
# Partially parsed test_pipe_with_nested_pipes. Retrieved 5/15 statements.
# Partially parsed test_pipe_with_empty_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_preserves_random_across_functions. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test-'
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 3
    var_1 = None

def test_case_0():
    var_0 = '!!'
    var_1 = 'hello'
    var_2 = None

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = 5
    var_4 = None

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = None


def test_case_0():
    var_0 = []
    var_1 = module_0.Random()
    var_2 = 'data'
    var_3 = var_0[0]
    var_4 = bool(var_0[0] is var_1)
    assert var_4 is True
    var_5 = var_0[1]
    var_6 = bool(var_0[1] is var_1)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pipe_applies_functions_in_sequence. Retrieved 1/7 statements.
# Partially parsed test_pipe_handles_key_func_with_random. Retrieved 2/9 statements.
# Partially parsed test_pipe_handles_mix_of_functions. Retrieved 3/12 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 2

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = 'John Doe'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pipe_key_function_with_random_parameter. Retrieved 3/19 statements.
# Partially parsed test_pipe_key_function_without_random_parameter. Retrieved 2/11 statements.
# Partially parsed test_pipe_key_function_mixed_parameters. Retrieved 3/13 statements.
# Partially parsed test_pipe_key_function_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_key_function_multiple_functions. Retrieved 1/7 statements.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'TEST-'
    var_2 = 'hello'
    var_3 = 'HELLO'

def test_case_0():
    var_0 = '!!!'
    var_1 = 'test'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'ab'
    var_2 = None

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_maybe_returns_value_with_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_returns_first_argument_with_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_with_probability_one. Retrieved 6/7 statements.
# Partially parsed test_maybe_with_different_value_types. Retrieved 7/8 statements.
# Partially parsed test_maybe_with_none_value. Retrieved 7/8 statements.
# Partially parsed test_maybe_with_complex_object. Retrieved 9/10 statements.


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


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 'special'
    var_3 = 0.7
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = module_1.maybe(var_2, var_1)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'special'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = -0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 1.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 123
    var_3 = 0.8
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 456
    var_6 = var_4(var_5, var_0)
    assert var_6 == 123


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = None
    var_3 = 0.6
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'not_none'
    var_6 = var_4(var_5, var_0)
    assert var_6 is None


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 0.9
    var_6 = module_1.maybe(var_4, var_5)
    var_7 = 'simple'
    var_8 = var_6(var_7, var_0)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True



# Parsed testcases at query #7
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


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 == 'missing'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 42
    var_6 = var_4(var_5)
    assert var_6 == 42


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = lambda x: x // var_0
    var_4 = None
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 3
    var_7 = var_5(var_6)
    assert var_7 == 3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pipe_key_function_with_random_parameter. Retrieved 2/10 statements.
# Partially parsed test_pipe_key_function_without_random_parameter. Retrieved 2/10 statements.
# Partially parsed test_pipe_key_function_mixed_parameters. Retrieved 5/16 statements.
# Partially parsed test_pipe_key_function_single_func. Retrieved 2/8 statements.
# Partially parsed test_pipe_key_function_no_random_passed. Retrieved 1/8 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = 2


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = 5



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_apply_if_condition_true_with_string. Retrieved 3/7 statements.
# Partially parsed test_apply_if_condition_false_with_string. Retrieved 3/7 statements.


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
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_1 = lambda x: len(x) > var_0
    var_2 = 'word'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 10
    var_6 = var_4(var_5)
    assert var_6 == 11


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 is None



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 3
    var_8 = var_6(var_7)
    assert var_8 == 4



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_regular_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_argument. Retrieved 2/9 statements.
# Partially parsed test_pipe_string_operations. Retrieved 2/8 statements.
# Partially parsed test_pipe_empty_functions. Retrieved 2/4 statements.


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

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = '  HELLO  '
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 42
    var_2 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pipe_key_handles_random_parameter_correctly. Retrieved 2/8 statements.



def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pipe_key_with_random_parameter. Retrieved 2/8 statements.



def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()



# Parsed testcases at query #14
#--------------------------




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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_apply_if_with_strings. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_strings_condition_false. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_strings_condition_false_with_otherwise. Retrieved 3/7 statements.



def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_2 = 'hi'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 == 'missing'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 42
    var_6 = var_4(var_5)
    assert var_6 == 42


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) > var_0 and sum(lst) > var_1
    var_3 = lambda lst: max(lst)
    var_4 = lambda lst: min(lst)
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 1
    var_7 = 3
    var_8 = 4
    var_9 = [var_6, var_0, var_7, var_8]
    var_10 = var_5(var_9)
    assert var_10 == 4


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) > var_0 and sum(lst) > var_1
    var_3 = lambda lst: max(lst)
    var_4 = lambda lst: min(lst)
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 1
    var_7 = [var_6, var_0]
    var_8 = var_5(var_7)
    assert var_8 == 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_string_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_random_parameter_ignored. Retrieved 2/8 statements.
# Partially parsed test_pipe_sequence_of_transformations. Retrieved 2/10 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 2


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'Hello'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = []
    var_2 = 'test'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'start_'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_functions_using_random. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_string_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_nested_pipe. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 5
    var_1 = None


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 6


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 6

def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = None

def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pipe_applies_functions_in_sequence. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_no_functions. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3
    var_2 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 4

def test_case_0():
    var_0 = []
    var_1 = 'test'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import mimesis.keys as module_0


def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    var_4 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


def test_case_0():
    var_0 = 'unsupported_algo'
    var_1 = module_0.hash_with(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    var_4 = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


def test_case_0():
    var_0 = 'md5'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    var_4 = '098f6bcd4621d373cade4e832627b4f6'
    var_5 = bool(var_3 == var_4)
    assert var_5 is True
    var_6 = 'sha512'
    var_7 = module_0.hash_with(var_6)
    var_8 = var_7(var_2)
    var_9 = 'ee26b0dd4af7e749aa1a8ee3c10ae9923f618980772e473f8819a5d4940e0db27ac185f8a0e1d5f84f88bc887fd67b143732c304cc5fa9ad8e6f57f50028a8ff'
    var_10 = bool(var_8 == var_9)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(var_4)
    assert var_5 == 'a, b, c'


def test_case_0():
    var_0 = ' | '
    var_1 = module_0.join(var_0)
    var_2 = 'pci'
    var_3 = 'promise'
    var_4 = 'excel'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1(var_5)
    assert var_6 == 'pci | promise | excel'


def test_case_0():
    var_0 = ';'
    var_1 = module_0.join(var_0)
    var_2 = []
    var_3 = var_1(var_2)
    assert var_3 == ''


def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 'single'
    var_3 = [var_2]
    var_4 = var_1(var_3)
    assert var_4 == 'single'


def test_case_0():
    var_0 = ' '
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1(var_5)
    assert var_6 == '1 2 3'


def test_case_0():
    var_0 = module_0.join()
    var_1 = 123
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = (var_2, var_3, var_4)
    var_6 = var_1(var_5)
    assert var_6 == 'a-b-c'


def test_case_0():
    var_0 = ','
    var_1 = module_0.join(var_0)
    var_2 = 3
    var_3 = range(var_2)
    var_4 = var_1(var_3)
    assert var_4 == '0,1,2'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_maybe_returns_value_with_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_returns_first_argument_with_probability. Retrieved 7/8 statements.
# Partially parsed test_maybe_probability_one. Retrieved 6/7 statements.
# Partially parsed test_maybe_with_different_value_types. Retrieved 7/8 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = 0.7
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'special'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 'special'
    var_3 = 0.7
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 'default'
    var_6 = var_4(var_5, var_0)
    assert var_6 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'special'
    var_2 = 0.0
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'default'
    var_6 = -0.5
    var_7 = module_1.maybe(var_1, var_6)
    var_8 = var_7(var_4, var_0)
    assert var_8 == 'default'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 'special'
    var_3 = module_1.maybe(var_2, var_1)
    var_4 = 'default'
    var_5 = var_3(var_4, var_0)
    assert var_5 == 'special'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 42
    var_3 = 0.5
    var_4 = module_1.maybe(var_2, var_3)
    var_5 = 100
    var_6 = var_4(var_5, var_0)
    assert var_6 == 42


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'other'
    var_5 = var_3(var_4, var_0)
    var_6 = bool(var_5 in ['value', 'other'])
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pipe_with_key_functions_using_random. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_key_functions_without_random. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_mixed_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_string_transformations. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_empty_functions. Retrieved 1/4 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/6 statements.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = '  HELLO  '

def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 4



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_romanize_returns_callable_for_supported_locale. Retrieved 1/7 statements.
# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.
# Partially parsed test_romanize_raises_type_error_when_non_string_passed_to_closure. Retrieved 1/7 statements.
# Partially parsed test_romanize_works_with_string_locale. Retrieved 4/6 statements.
# Partially parsed test_romanize_translates_common_letters. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0


def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'
    var_3 = var_1(var_2)


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)


def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 'ё'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_string_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_mixed_function_signatures. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_nested_pipe. Retrieved 1/6 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 5/11 statements.
# Partially parsed test_pipe_with_list_operations. Retrieved 3/11 statements.
# Partially parsed test_pipe_with_dict_operations. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 3

def test_case_0():
    var_0 = '  HELLO  '

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 3


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 3

def test_case_0():
    var_0 = []
    var_1 = 42

def test_case_0():
    var_0 = 1

import mimesis.keys as module_0


def test_case_0():
    var_0 = 'pre-'
    var_1 = module_0.prefix(var_0)
    var_2 = '-suf'
    var_3 = module_0.suffix(var_2)
    var_4 = [var_1, var_3]
    var_5 = 'test'

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'old'
    var_1 = 'data'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------





def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 == 'hello'


def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello world'
    var_3 = var_1(var_2)
    assert var_3 == 'he...'


def test_case_0():
    var_0 = 5
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello world'
    var_4 = var_2(var_3)
    assert var_4 == 'hel!!'


def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 == 'hello'


def test_case_0():
    var_0 = 5
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'hello'


def test_case_0():
    var_0 = 2
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == '..'


def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''


def test_case_0():
    var_0 = -1
    var_1 = module_0.truncate(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pipe_key_function_with_random_parameter. Retrieved 4/18 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = 'TEST-'
    var_1 = module_0.Random()
    var_2 = 2
    var_3 = 'hello'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 4/15 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 4/13 statements.
# Partially parsed test_pipe_single_function. Retrieved 3/7 statements.
# Partially parsed test_pipe_no_functions. Retrieved 3/5 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/9 statements.
# Partially parsed test_pipe_chain_with_and_without_random. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_'
    var_1 = 'hello'
    var_2 = None
    var_3 = 'TEST_HELLOTEST_HELLO'

def test_case_0():
    var_0 = '!'
    var_1 = 'hi'
    var_2 = None
    var_3 = 'hihi!'

def test_case_0():
    var_0 = 5
    var_1 = None
    var_2 = 6

def test_case_0():
    var_0 = []
    var_1 = 'anything'
    var_2 = None
    var_3 = 'anything'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 7
    var_2 = 'num'
    var_3 = 'num7'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3
    var_2 = 'test'
    var_3 = 'TEST3'



# Parsed testcases at query #10
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


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_2 = 'word'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'HI'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'transformed'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 == 'transformed'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'transformed'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 42
    var_6 = var_4(var_5)
    assert var_6 == 42


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) > var_0 and sum(lst) > var_1
    var_3 = lambda lst: max(lst)
    var_4 = lambda lst: min(lst)
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 1
    var_7 = 3
    var_8 = 4
    var_9 = [var_6, var_0, var_7, var_8]
    var_10 = var_5(var_9)
    assert var_10 == 4


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) > var_0 and sum(lst) > var_1
    var_3 = lambda lst: max(lst)
    var_4 = lambda lst: min(lst)
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 1
    var_7 = [var_6, var_0]
    var_8 = var_5(var_7)
    assert var_8 == 1



# Parsed testcases at query #11
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 5


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



# Parsed testcases at query #12
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = lambda x: x - var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 4
    var_8 = lambda x: var_0
    var_9 = lambda x: x.upper()
    var_10 = module_0.apply_if(var_8, var_9)
    var_11 = 'test'
    var_12 = var_10(var_11)
    assert var_12 == 'test'
    var_13 = lambda x: var_0
    var_14 = 2
    var_15 = lambda x: x * var_14
    var_16 = None
    var_17 = module_0.apply_if(var_13, var_15, var_16)
    var_18 = 3
    var_19 = var_17(var_18)
    assert var_19 == 3



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 5


def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 10
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 15



# Parsed testcases at query #14
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = lambda x: x - var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 4
    var_8 = lambda x: var_0
    var_9 = lambda x: x.upper()
    var_10 = module_0.apply_if(var_8, var_9)
    var_11 = 'test'
    var_12 = var_10(var_11)
    assert var_12 == 'test'
    var_13 = lambda x: var_0
    var_14 = 2
    var_15 = lambda x: x * var_14
    var_16 = None
    var_17 = module_0.apply_if(var_13, var_15, var_16)
    var_18 = 3
    var_19 = var_17(var_18)
    assert var_19 == 3



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pipe_with_key_functions_using_random. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_key_functions_without_random. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_mixed_key_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_string_transformations. Retrieved 1/8 statements.
# Partially parsed test_pipe_with_empty_functions. Retrieved 1/4 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/6 statements.
# Partially parsed test_pipe_preserves_random_instance. Retrieved 3/11 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 5


def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 4


def test_case_0():
    var_0 = module_0.Random()
    var_1 = None
    var_2 = 10
    var_3 = bool(var_1 is var_0)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
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


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 'transformed'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 'input'
    var_6 = var_4(var_5)
    assert var_6 == 'input'


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x + var_0
    var_3 = lambda x: x - var_0
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 10
    var_6 = var_4(var_5)
    assert var_6 == 11


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) == var_0 and sum(lst) > var_1
    var_3 = lambda lst: sum(lst)
    var_4 = module_0.apply_if(var_2, var_3)
    var_5 = 6
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = var_4(var_7)
    assert var_8 == 11


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = lambda lst: len(lst) == var_0 and sum(lst) > var_1
    var_3 = lambda lst: sum(lst)
    var_4 = module_0.apply_if(var_2, var_3)
    var_5 = 1
    var_6 = [var_5, var_0]
    var_7 = var_4(var_6)
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_string_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/8 statements.
# Partially parsed test_pipe_mixed_with_and_without_random. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_nested_pipe. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10

def test_case_0():
    var_0 = []
    var_1 = 42

def test_case_0():
    var_0 = 5



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_and_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_none_otherwise_and_condition_false. Retrieved 3/6 statements.


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


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'not an int'


def test_case_0():
    var_0 = lambda lst: len(lst) == sum(lst)
    var_1 = 2
    var_2 = lambda lst: [x * var_1 for x in lst]
    var_3 = -1
    var_4 = lambda lst: lst[::var_3]
    var_5 = module_0.apply_if(var_0, var_2, var_4)
    var_6 = 1
    var_7 = [var_6, var_6, var_6]
    var_8 = var_5(var_7)
    var_9 = bool(var_8 == [2, 2, 2])
    assert var_9 is True


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = lambda x: x // var_0
    var_4 = None
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 4
    var_7 = var_5(var_6)
    assert var_7 == 2
    var_8 = 3
    var_9 = var_5(var_8)
    assert var_9 == 3



# Parsed testcases at query #19
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = lambda x: x - var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 4



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_argument. Retrieved 3/8 statements.
# Partially parsed test_pipe_chain_with_ignored_random. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 'test'
    var_1 = None

def test_case_0():
    var_0 = 4
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'anything'
    var_2 = None

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10

def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pipe_key_function_with_random_parameter. Retrieved 4/18 statements.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = 'TEST_'
    var_3 = 5
    var_4 = 1



# Parsed testcases at query #22
#--------------------------




import mimesis.keys as module_0


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = -1
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 10



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_without_otherwise. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_with_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_condition_true_with_complex_transform. Retrieved 4/8 statements.
# Partially parsed test_apply_if_condition_false_with_complex_otherwise. Retrieved 2/6 statements.



def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_1 = lambda x: len(x) > var_0
    var_2 = 'word'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'HI'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'None'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3, var_0)
    var_5 = 42
    var_6 = var_4(var_5)
    assert var_6 == 42

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda x: len(str(x))
    var_1 = 12345



# Parsed testcases at query #24
#--------------------------





def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = -1
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 10



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_apply_if_with_string_condition_true. Retrieved 3/6 statements.
# Partially parsed test_apply_if_with_string_condition_false_and_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_none_otherwise. Retrieved 3/6 statements.



def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'string'


def test_case_0():
    var_0 = 2
    var_1 = lambda lst: len(lst) > var_0
    var_2 = lambda lst: sum(lst)
    var_3 = lambda lst: max(lst)
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 1
    var_6 = [var_5, var_0]
    var_7 = var_4(var_6)
    assert var_7 == 2


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = lambda x: x // var_0
    var_4 = lambda x: x * var_0
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 4
    var_7 = var_5(var_6)
    assert var_7 == 2


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = lambda x: x // var_0
    var_4 = lambda x: x * var_0
    var_5 = module_0.apply_if(var_2, var_3, var_4)
    var_6 = 3
    var_7 = var_5(var_6)
    assert var_7 == 6



# Parsed testcases at query #26
#--------------------------





def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = -1
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 10



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pipe_handles_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'John_Doe'
    var_1 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pipe_with_key_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_argument. Retrieved 3/8 statements.
# Partially parsed test_pipe_with_three_functions. Retrieved 2/10 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_list_operations. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 'test'
    var_1 = None

def test_case_0():
    var_0 = 4
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'anything'
    var_2 = None

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10

def test_case_0():
    var_0 = 5
    var_1 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'start'
    var_1 = [var_0]
    var_2 = None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_apply_if_condition_true. Retrieved 3/6 statements.


import mimesis.keys as module_0


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = -1
    var_5 = lambda x: x * var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = 5
    var_8 = var_6(var_7)
    assert var_8 == 10
    var_9 = -3
    var_10 = var_6(var_9)
    assert var_10 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = lambda x: x.startswith(var_0)
    var_2 = lambda x: x.upper()
    var_3 = module_0.apply_if(var_1, var_2)
    var_4 = 'apple'
    var_5 = var_3(var_4)
    assert var_5 == 'APPLE'
    var_6 = 'banana'
    var_7 = var_3(var_6)
    assert var_7 == 'banana'

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 42


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = lambda x: x * var_0
    var_4 = module_0.apply_if(var_2, var_3)
    var_5 = 3
    var_6 = var_4(var_5)
    assert var_6 == 3


def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 'big'
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 15
    var_7 = var_5(var_6)
    assert var_7 == 'big'
    var_8 = 5
    var_9 = var_5(var_8)
    assert var_9 == 5



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pipe_handles_key_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pipe_handles_key_functions_without_random_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false_with_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_complex_condition. Retrieved 8/11 statements.
# Partially parsed test_apply_if_with_complex_condition_false. Retrieved 10/13 statements.



def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5


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
    var_2 = 'word'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'hi'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 == 'missing'


def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'missing'
    var_3 = lambda x: var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 42
    var_6 = var_4(var_5)
    assert var_6 == 42

def test_case_0():
    var_0 = 0
    var_1 = 'empty'
    var_2 = [var_1]
    var_3 = lambda x: var_2
    var_4 = 1
    var_5 = [var_4]
    var_6 = lambda x: x + var_5
    var_7 = []

def test_case_0():
    var_0 = 0
    var_1 = 'empty'
    var_2 = [var_1]
    var_3 = lambda x: var_2
    var_4 = 1
    var_5 = [var_4]
    var_6 = lambda x: x + var_5
    var_7 = 2
    var_8 = 3
    var_9 = [var_7, var_8]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pipe_key_with_random_parameter. Retrieved 2/8 statements.
# Partially parsed test_pipe_key_without_random_parameter. Retrieved 1/7 statements.
# Partially parsed test_pipe_key_mixed_functions. Retrieved 3/10 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Random()

def test_case_0():
    var_0 = 'hello'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = '-suf'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pipe_with_random_parameter. Retrieved 2/8 statements.



def test_case_0():
    var_0 = 5
    var_1 = module_0.Random()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pipe_applies_functions_in_sequence. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_string_operations. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_random_parameter_optional. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_nested_pipes. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = '  hello-world  '


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 3
    var_2 = 4

def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 0



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pipe_key_function_handles_type_error_correctly. Retrieved 2/8 statements.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'



