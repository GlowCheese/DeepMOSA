####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.maybe(var_0)
    var_2 = module_1.Random()
    var_3 = 'original'
    var_4 = var_1(var_3, var_2)
    var_5 = bool(var_4 in ['original', 'test_value'])
    assert var_5 is True

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'test_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original'



# Parsed testcases at query #2
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'any_value'
    var_2 = var_0(var_1)
    assert var_2 == '[REDACTED]'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'any_value'
    var_3 = var_1(var_2)
    assert var_3 == '[CLASSIFIED]'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'input_value'
    var_2 = var_0(var_1)
    assert var_2 == '[REDACTED]'
    var_3 = module_0.redact()
    var_4 = None
    var_5 = var_3(var_4)
    assert var_5 == '[REDACTED]'
    var_6 = module_0.redact()
    var_7 = 123
    var_8 = var_6(var_7)
    assert var_8 == '[REDACTED]'



# Parsed testcases at query #3
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
    var_0 = module_0.join()
    var_1 = []
    var_2 = var_0(var_1)
    assert var_2 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'hello'
    var_2 = [var_1]
    var_3 = var_0(var_2)
    assert var_3 == 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '-'
    var_1 = module_0.join(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1(var_5)
    assert var_6 == '1-2-3'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'not a list'
    var_2 = var_0(var_1)



# Parsed testcases at query #4
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = var_0(var_1)
    assert var_2 == '<test>'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '['
    var_1 = ']'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'
    var_4 = var_2(var_3)
    assert var_4 == '[test]'

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 123
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 == 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello world'
    var_3 = var_1(var_2)
    assert var_3 == 'hello wor...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '!!'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'hello world'
    var_4 = var_2(var_3)
    assert var_4 == 'hello wor!!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'hello'
    var_3 = var_1(var_2)
    assert var_3 == 'hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_apply_if_with_string_condition_and_transform. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_and_otherwise. Retrieved 3/7 statements.


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
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = -5
    var_8 = var_6(var_7)
    assert var_8 == -4

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

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/5 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_input. Retrieved 1/5 statements.
# Partially parsed test_romanize_with_valid_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'тест'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 4/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/12 statements.
# Partially parsed test_pipe_handles_function_without_random. Retrieved 4/6 statements.
# Partially parsed test_pipe_returns_original_value_when_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'hello'
    var_2 = 'HELLO'
    var_3 = 5

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = None

def test_case_0():
    var_0 = []
    var_1 = 'hello'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '!'
    var_2 = lambda x: x + var_1
    var_3 = 'hello'
    var_4 = 'HELLO'
    var_5 = '!!'
    var_6 = '?!'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_romanize_with_unsupported_locale.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #13
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x.upper()
    var_3 = module_0.apply_if(var_1, var_2)
    var_4 = 'test'
    var_5 = var_3(var_4)
    assert var_5 == 'test'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/12 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Привіт'
    var_2 = 'Сәлем'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'
    var_3 = var_1(var_2)
    assert var_3 == 'Privet'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------

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
    var_4 = 0
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 0

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 is None



# Parsed testcases at query #18
#--------------------------

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
    var_4 = 0
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 0

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/12 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'
    var_1 = 'привіт'
    var_2 = 'сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'привет'
    var_3 = var_1(var_2)
    assert var_3 == 'privet'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pipe_function_docstring. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #23
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 2/6 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Мир'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'
    var_3 = var_1(var_2)
    assert var_3 == 'Pryvit'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 10

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = -5
    var_7 = var_5(var_6)
    assert var_7 == -5

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
    var_5 = var_4(var_0)
    assert var_5 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.


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
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_5 = var_4(var_0)
    assert var_5 is None

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 6



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/12 statements.
# Partially parsed test_pipe_with_function_without_random. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 5
    var_3 = 1
    var_4 = 10

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = []
    var_1 = 42

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 5
    var_3 = 2
    var_4 = var_2 * var_3
    var_5 = 1
    var_6 = 10



# Parsed testcases at query #30
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

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
    var_4 = 0
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 0

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = var_4(var_0)
    assert var_5 is None



# Parsed testcases at query #3
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'name'
    var_3 = var_1(var_2)
    assert var_3 == 'user_name'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 6/18 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Мир'
    var_2 = 'Привіт'
    var_3 = 'Світ'
    var_4 = 'Сәлем'
    var_5 = 'Әлем'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привет'
    var_3 = var_1(var_2)
    assert var_3 == 'Privet'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_romanize_with_unsupported_locale.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_maybe_closure_returns_mixed_values. Retrieved 7/8 statements.
# Partially parsed test_maybe_closure_returns_original_value_when_random_high. Retrieved 7/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 0
    var_5 = var_2(var_4, var_3)
    assert var_5 == 42

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 100
    var_5 = var_2(var_4, var_3)
    assert var_5 == 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 0.25
    var_5 = 100
    var_6 = var_2(var_5, var_3)
    assert var_6 == 42

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 0.75
    var_5 = 100
    var_6 = var_2(var_5, var_3)
    assert var_6 == 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 100
    var_5 = var_2(var_4, var_3)
    assert var_5 == 100

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 100
    var_5 = var_2(var_4, var_3)
    assert var_5 == 100



# Parsed testcases at query #7
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = callable(var_0)
    var_2 = bool(var_1)
    assert var_2 is True
    var_3 = 'password'
    var_4 = var_0(var_3)
    assert var_4 == '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha1'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'password'
    var_3 = var_1(var_2)
    assert var_3 == 'd3e7130d657733468b10c1fd207c4d62b7180cda'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'unsupported'
    var_1 = module_0.hash_with(var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 123
    var_2 = var_0(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/8 statements.
# Partially parsed test_pipe_with_exception_handling. Retrieved 2/8 statements.
# Partially parsed test_pipe_empty_input. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_custom_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_lambda_functions. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'HELLO'
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
    var_4 = [var_3]
    var_5 = 'Number: '

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5

def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 3
    var_3 = lambda x: x + var_2
    var_4 = [var_1, var_3]
    var_5 = 5
    var_6 = None



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #10
#--------------------------

# Failed to parse test_romanize_locale_validation.




# Parsed testcases at query #11
#--------------------------




import mimesis.keys as module_0

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



# Parsed testcases at query #12
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'
    var_3 = var_1(var_2)
    assert var_3 == 'Hello'
    var_4 = 'Hello World'
    var_5 = var_1(var_4)
    assert var_5 == 'Hello W...'
    var_6 = 'Hello World!'
    var_7 = var_1(var_6)
    assert var_7 == 'Hello W...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Hello World'
    var_4 = var_2(var_3)
    assert var_4 == 'Hello W...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'
    var_3 = var_1(var_2)
    assert var_3 == 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Москва'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Київ'
    var_3 = var_1(var_2)
    assert var_3 == 'Kyiv'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_input_type. Retrieved 1/5 statements.
# Failed to parse test_romanize_with_unsupported_locale.


def test_case_0():
    var_0 = 'привет'

def test_case_0():
    var_0 = 123

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'kk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'қазақ'
    var_3 = var_1(var_2)
    assert var_3 == 'qazaq'

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.romanize(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pipe_function_docstring. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #18
#--------------------------




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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.


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
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_5 = var_4(var_0)
    assert var_5 is None

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 'default'
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = var_5(var_0)
    assert var_6 == 'default'



# Parsed testcases at query #20
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = None
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 5



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'привет'
    var_3 = var_1(var_2)
    assert var_3 == 'privet'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_correct_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #23
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x.upper()
    var_3 = lambda x: x.lower()
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/12 statements.
# Partially parsed test_pipe_handles_function_without_random. Retrieved 2/7 statements.
# Partially parsed test_pipe_with_empty_input. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_none_random. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'Hello'
    var_2 = 'hello'
    var_3 = -1

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

def test_case_0():
    var_0 = ''
    var_1 = None

def test_case_0():
    var_0 = 'HELLO'
    var_1 = None



# Parsed testcases at query #25
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #26
#--------------------------




import mimesis.locales as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.validate_locale(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_correct_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/12 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'привет'
    var_1 = 'привіт'
    var_2 = 'сәлем'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.romanize(var_0)
    var_2 = 'привет'
    var_3 = var_1(var_2)
    assert var_3 == 'privet'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.romanize(var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #29
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x.upper()
    var_3 = module_0.apply_if(var_1, var_2)
    var_4 = 'test'
    var_5 = var_3(var_4)
    assert var_5 == 'test'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 1/4 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'uk'
    var_1 = module_0.romanize(var_0)
    var_2 = 'Привіт'
    var_3 = var_1(var_2)
    assert var_3 == 'Privit'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.romanize(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/12 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'HELLO'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'Hello'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'HELLO'
    var_2 = 'hello'
    var_3 = 5

def test_case_0():
    var_0 = 'test'
    var_1 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/6 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/11 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/8 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_function_that_ignores_random. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = None

def test_case_0():
    var_0 = 'test'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = 'test_'
    var_3 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'abc'

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = 'input'
    var_1 = module_0.Random()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pipe_predicate_false. Retrieved 3/6 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user-'
    var_1 = module_0.prefix(var_0)
    var_2 = 'user-john-doe'



# Parsed testcases at query #34
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 2/4 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 4/12 statements.
# Partially parsed test_pipe_with_function_raising_type_error. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'hello'
    var_2 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'HELLO'
    var_2 = 'hello'
    var_3 = 5

def test_case_0():
    var_0 = 'hello'
    var_1 = None



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_pipe_predicate_false.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pipe_predicate_false. Retrieved 4/8 statements.


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user-'
    var_1 = module_0.prefix(var_0)
    var_2 = 'John Doe'
    var_3 = 'user-john-doe'



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_pipe_predicate_false.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_apply_if_with_string_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_false. Retrieved 3/7 statements.


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
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -5
    var_6 = var_4(var_5)
    assert var_6 == -5

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
    var_5 = var_4(var_0)
    assert var_5 is None

import mimesis.keys as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 6



# Parsed testcases at query #41
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'test'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_apply_if_with_string_condition_and_transform. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_condition_and_otherwise. Retrieved 3/7 statements.


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
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #43
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = None
    var_6 = var_4(var_5)
    assert var_6 is None



# Parsed testcases at query #44
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = module_0.apply_if(var_1, var_2, var_3)
    var_5 = None
    var_6 = var_4(var_5)
    assert var_6 is None



