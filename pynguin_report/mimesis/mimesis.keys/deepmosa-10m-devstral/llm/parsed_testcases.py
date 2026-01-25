####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_romanize_with_russian_locale. Retrieved 2/6 statements.
# Partially parsed test_romanize_with_ukrainian_locale. Retrieved 2/6 statements.
# Partially parsed test_romanize_with_kazakh_locale. Retrieved 2/6 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_invalid_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Москва'

def test_case_0():
    var_0 = 'Привіт'
    var_1 = 'Київ'

def test_case_0():
    var_0 = 'Сәлем'
    var_1 = 'Астана'

def test_case_0():
    var_0 = 123

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



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_romanize_raises_value_error_for_unsupported_locale.




# Parsed testcases at query #3
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'short'
    var_3 = var_1(var_2)
    assert var_3 == 'short'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = '1234567890'
    var_3 = var_1(var_2)
    assert var_3 == '1234567890'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = '1234567890123'
    var_3 = var_1(var_2)
    assert var_3 == '1234567...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = '1234567890123'
    var_4 = var_2(var_3)
    assert var_4 == '1234567...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 123
    var_3 = var_1(var_2)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.truncate(var_0)

import mimesis.keys as module_0

def test_case_0():
    var_0 = -5
    var_1 = module_0.truncate(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 2/7 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 3/10 statements.
# Partially parsed test_pipe_with_no_random_parameter. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 3/11 statements.


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

def test_case_0():
    var_0 = 'test'
    var_1 = None

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 'hello'



# Parsed testcases at query #5
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
    var_2 = 'valid'
    var_3 = lambda x: var_2
    var_4 = 'invalid'
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 'invalid'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_apply_if_works_with_strings. Retrieved 3/7 statements.
# Partially parsed test_apply_if_works_with_strings_and_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_works_with_custom_objects. Retrieved 7/12 statements.
# Partially parsed test_apply_if_works_with_custom_objects_and_otherwise. Retrieved 6/11 statements.


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
    var_0 = 2
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = lambda x: x + var_3
    var_5 = 0
    var_6 = [var_5]
    var_7 = lambda x: x + var_6
    var_8 = module_0.apply_if(var_1, var_4, var_7)
    var_9 = [var_2, var_0]
    var_10 = var_8(var_9)
    var_11 = bool(var_10 == [1, 2, 1])
    assert var_11 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = lambda x: x + var_3
    var_5 = module_0.apply_if(var_1, var_4)
    var_6 = [var_2]
    var_7 = var_5(var_6)
    var_8 = bool(var_7 == [1])
    assert var_8 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = lambda x: x.value > var_1
    var_3 = 2
    var_4 = lambda x: TestClass(x.value * var_3)
    var_5 = lambda x: TestClass(x.value)
    var_6 = module_0.apply_if(var_2, var_4, var_5)

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = 5
    var_2 = lambda x: x.value > var_1
    var_3 = lambda x: TestClass(x.value * var_0)
    var_4 = lambda x: TestClass(x.value)
    var_5 = module_0.apply_if(var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pipe_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #9
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
    var_1 = 0.7
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    var_6 = bool(var_5 in ['original_value', 'test_value'])
    assert var_6 is True

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'test_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/9 statements.
# Partially parsed test_pipe_handles_type_error_gracefully. Retrieved 2/6 statements.
# Partially parsed test_pipe_returns_original_value_when_no_functions. Retrieved 2/4 statements.


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
    var_4 = [var_3]
    var_5 = 'value'

import mimesis.random as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Random()

def test_case_0():
    var_0 = []
    var_1 = 'unchanged'
    var_2 = None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'password'
    var_3 = var_1(var_2)
    assert var_3 == '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'

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

import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 123
    var_3 = var_1(var_2)



# Parsed testcases at query #2
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'
    var_3 = var_1(var_2)
    assert var_3 == 'example.io'

import mimesis.keys as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.suffix(var_0)
    var_2 = 'test'
    var_3 = var_1(var_2)
    assert var_3 == 'test'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pipe_empty. Retrieved 1/3 statements.
# Partially parsed test_pipe_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random. Retrieved 4/11 statements.
# Partially parsed test_pipe_with_exception_handling. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = 'test_'
    var_3 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'



# Parsed testcases at query #4
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'
    var_3 = var_1(var_2)
    assert var_3 == 'Hello, W...'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 13
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello, World!'
    var_3 = var_1(var_2)
    assert var_3 == 'Hello, World!'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.truncate(var_0)
    var_2 = 'Hello'
    var_3 = var_1(var_2)
    assert var_3 == 'Hello'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 8
    var_1 = '..'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'Hello, World!'
    var_4 = var_2(var_3)
    assert var_4 == 'Hello..'

import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    assert var_3 == ''

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_romanize_with_valid_locale. Retrieved 3/12 statements.
# Failed to parse test_romanize_with_invalid_locale.
# Partially parsed test_romanize_with_non_string_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Привіт'
    var_2 = 'Сәлем'

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

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_apply_if_works_with_strings. Retrieved 3/7 statements.
# Partially parsed test_apply_if_works_with_strings_and_false_condition. Retrieved 3/7 statements.
# Partially parsed test_apply_if_works_with_none_otherwise. Retrieved 3/6 statements.


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
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = -3
    var_8 = var_6(var_7)
    assert var_8 == -2

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = -3
    var_6 = var_4(var_5)
    assert var_6 == -3

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hello'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #9
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
    var_1 = 0.7
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    var_6 = bool(var_5 in ['original_value', 'test_value'])
    assert var_6 is True

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.0
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'test_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = -0.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'

import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1.5
    var_2 = module_0.maybe(var_0, var_1)
    var_3 = module_1.Random()
    var_4 = 'original_value'
    var_5 = var_2(var_4, var_3)
    assert var_5 == 'original_value'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_apply_if_string_condition_true. Retrieved 3/7 statements.
# Partially parsed test_apply_if_string_condition_false_with_otherwise. Retrieved 3/7 statements.
# Partially parsed test_apply_if_string_condition_false_without_otherwise. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = 'hi'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pipe_empty_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_random. Retrieved 6/12 statements.
# Partially parsed test_pipe_with_exception_handling. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = 'test_'
    var_3 = 1
    var_4 = '_'
    var_5 = result.split(var_4)[var_3]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pipe_docstring_starts_with_pipe_multiple_key_functions_together. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pipe_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Pipe multiple key functions together.'



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_apply_if_with_string. Retrieved 3/7 statements.
# Partially parsed test_apply_if_with_string_and_false_condition. Retrieved 3/7 statements.


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

import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x != var_0
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = lambda x: x - var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = var_5(var_0)
    assert var_6 == -1

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
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = None
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = -5
    var_7 = var_5(var_6)
    assert var_7 == -5



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------




import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = lambda x: x / var_2
    var_5 = module_0.apply_if(var_1, var_3, var_4)
    var_6 = 5
    var_7 = var_5(var_6)
    assert var_7 == 10



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_apply_if_with_string_condition_and_transform. Retrieved 3/6 statements.
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

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = lambda x: x + var_3
    var_5 = module_0.apply_if(var_1, var_4)
    var_6 = 3
    var_7 = [var_2, var_0, var_6]
    var_8 = var_5(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 1])
    assert var_9 is True

import mimesis.keys as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = lambda x: x + var_3
    var_5 = 0
    var_6 = [var_5]
    var_7 = lambda x: x + var_6
    var_8 = module_0.apply_if(var_1, var_4, var_7)
    var_9 = [var_2, var_0]
    var_10 = var_8(var_9)
    var_11 = bool(var_10 == [1, 2, 0])
    assert var_11 is True



# Parsed testcases at query #26
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
    var_7 = 1
    var_8 = var_6(var_7)
    assert var_8 == 2



# Parsed testcases at query #27
#--------------------------




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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_apply_if_with_string_condition_and_transform. Retrieved 3/6 statements.
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
    var_4 = 0
    var_5 = lambda x: var_4
    var_6 = module_0.apply_if(var_1, var_3, var_5)
    var_7 = var_6(var_0)
    assert var_7 == 0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pipe_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_multiple_functions. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_random. Retrieved 4/12 statements.
# Partially parsed test_pipe_empty_input. Retrieved 2/5 statements.
# Partially parsed test_pipe_with_none. Retrieved 1/4 statements.
# Partially parsed test_pipe_with_exception_handling. Retrieved 2/6 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 2/7 statements.


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
    var_3 = 5

def test_case_0():
    var_0 = ''
    var_1 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'hello'
    var_1 = None

def test_case_0():
    var_0 = 'Hello'
    var_1 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pipe_applies_single_function. Retrieved 2/5 statements.
# Partially parsed test_pipe_applies_multiple_functions. Retrieved 4/7 statements.
# Partially parsed test_pipe_handles_none_random. Retrieved 3/5 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/9 statements.
# Partially parsed test_pipe_empty_functions_list. Retrieved 2/4 statements.


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
    var_2 = [var_1]
    var_3 = 'test'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 100
    var_3 = lambda x, r: x + str(r.randint(var_1, var_2))
    var_4 = [var_3]
    var_5 = 'value'

def test_case_0():
    var_0 = []
    var_1 = 'unchanged'
    var_2 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pipe_with_single_function. Retrieved 1/5 statements.
# Partially parsed test_pipe_with_multiple_functions. Retrieved 1/7 statements.
# Partially parsed test_pipe_with_no_functions. Retrieved 1/3 statements.
# Partially parsed test_pipe_with_random_parameter. Retrieved 5/12 statements.
# Partially parsed test_pipe_with_mixed_functions. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = []
    var_1 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 5
    var_3 = 1
    var_4 = 10

def test_case_0():
    var_0 = 5



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_pipe_predicate_false.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_pipe_predicate_false.




