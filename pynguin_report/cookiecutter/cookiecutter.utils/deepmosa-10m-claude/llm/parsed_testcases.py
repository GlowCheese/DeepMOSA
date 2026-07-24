####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_multiple_filters. Retrieved 8/17 statements.
# Partially parsed test_simple_filter_preserves_filter_function. Retrieved 9/14 statements.
# Partially parsed test_simple_filter_with_complex_function. Retrieved 8/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_filter'
    var_2 = bool('my_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_filter']
    var_4 = 'my_filter'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'first_filter'
    var_2 = bool('first_filter' in var_0.filters)
    assert var_2 is True
    var_3 = 'second_filter'
    var_4 = bool('second_filter' in var_0.filters)
    assert var_4 is True
    var_5 = 'first_filter'
    var_6 = var_0.filters[var_5]
    var_7 = 'test'
    var_8 = var_6(var_7)
    assert var_8 == 'test_first'
    var_9 = 'second_filter'
    var_10 = var_0.filters[var_9]
    var_11 = var_10(var_7)
    assert var_11 == 'test_second'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom_filter'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = '!'
    var_5 = var_2(var_3, suffix=var_4)
    assert var_5 == 'hello!'
    var_6 = var_0.filters[var_1]
    var_7 = 'world'
    var_8 = var_6(var_7)
    assert var_8 == 'world'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_filter'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'
    var_5 = var_0.filters[var_1]
    var_6 = '12345'
    var_7 = var_5(var_6)
    assert var_7 == '54321'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Failed to parse test_simple_filter_registers_filter_in_environment.
# Partially parsed test_simple_filter_filter_works_in_jinja_template. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_with_multiple_filters. Retrieved 3/16 statements.
# Failed to parse test_simple_filter_preserves_function_name.
# Partially parsed test_simple_filter_with_lambda. Retrieved 9/11 statements.


def test_case_0():
    var_0 = '{{ text|reverse_string }}'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'add_prefix'
    var_1 = 'add_suffix'
    var_2 = 'add_prefix'
    var_3 = 'test'
    var_4 = 'add_suffix'

import cookiecutter.utils as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = module_0.simple_filter(var_1)
    var_3 = [var_2]
    var_4 = module_1.Environment(extensions=var_3)
    var_5 = 'double'
    var_6 = bool('double' in var_4.filters)
    assert var_6 is True
    var_7 = 'double'
    var_8 = var_4.filters[var_7]
    var_9 = 5
    var_10 = var_8(var_9)
    assert var_10 == 10



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_filter.
# Failed to parse test_simple_filter_returns_extension_class.
# Failed to parse test_filter.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/2 statements.
# Failed to parse test_simple_filter_extension_name_matches_function_name.
# Partially parsed test_simple_filter_filter_works_correctly. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_multiple_filters. Retrieved 8/17 statements.
# Partially parsed test_simple_filter_preserves_function_behavior. Retrieved 6/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'filter1'
    var_2 = var_0.filters[var_1]
    var_3 = 'test'
    var_4 = var_2(var_3)
    assert var_4 == 'test_1'
    var_5 = 'filter2'
    var_6 = var_0.filters[var_5]
    var_7 = var_6(var_3)
    assert var_7 == 'test_2'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_prefix'
    var_2 = var_0.filters[var_1]
    var_3 = 'test'
    var_4 = 'custom_'
    var_5 = var_2(var_3, prefix=var_4)
    assert var_5 == 'custom_test'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_filter.
# Failed to parse test_simple_filter_returns_extension_class.
# Failed to parse test_filter.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/2 statements.
# Failed to parse test_simple_filter_extension_name_matches_function_name.
# Partially parsed test_simple_filter_filter_works_in_template. Retrieved 4/10 statements.
# Partially parsed test_simple_filter_with_multiple_filters. Retrieved 8/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = '{{ text | reverse_string }}'
    var_2 = var_0.from_string(var_1)
    var_3 = 'hello'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'filter_one'
    var_2 = bool('filter_one' in var_0.filters)
    assert var_2 is True
    var_3 = 'filter_two'
    var_4 = bool('filter_two' in var_0.filters)
    assert var_4 is True
    var_5 = 'filter_one'
    var_6 = var_0.filters[var_5]
    var_7 = 'test'
    var_8 = var_6(var_7)
    assert var_8 == 'test_one'
    var_9 = 'filter_two'
    var_10 = var_0.filters[var_9]
    var_11 = var_10(var_7)
    assert var_11 == 'test_two'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Failed to parse test_simple_filter_registers_filter_in_environment.
# Partially parsed test_simple_filter_filter_function_works. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_with_multiple_arguments. Retrieved 3/11 statements.
# Partially parsed test_simple_filter_extension_initialization. Retrieved 1/1 statements.


def test_case_0():
    var_0 = 'reverse_string'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'add_numbers'
    var_1 = 5
    var_2 = 3

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_with_different_function_names. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_filter_works_correctly. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_filters. Retrieved 8/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_filter'
    var_2 = bool('my_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_filter']

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom_transform'
    var_2 = bool('custom_transform' in var_0.filters)
    assert var_2 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'filter1'
    var_2 = bool('filter1' in var_0.filters)
    assert var_2 is True
    var_3 = 'filter2'
    var_4 = bool('filter2' in var_0.filters)
    assert var_4 is True
    var_5 = 'filter1'
    var_6 = var_0.filters[var_5]
    var_7 = 'test'
    var_8 = var_6(var_7)
    assert var_8 == 'test1'
    var_9 = 'filter2'
    var_10 = var_0.filters[var_9]
    var_11 = var_10(var_7)
    assert var_11 == 'test2'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Failed to parse test_simple_filter_registers_filter_in_environment.
# Partially parsed test_simple_filter_filter_works_correctly. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_with_multiple_arguments. Retrieved 3/11 statements.
# Failed to parse test_simple_filter_preserves_function_name.
# Partially parsed test_simple_filter_extension_initialization. Retrieved 1/1 statements.


def test_case_0():
    var_0 = '{{ text|reverse_filter }}'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'multiply_filter'
    var_1 = 3
    var_2 = 4

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Failed to parse test_simple_filter_registers_filter_in_environment.
# Partially parsed test_simple_filter_filter_works_correctly. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_with_multiple_arguments. Retrieved 2/10 statements.
# Failed to parse test_filter.
# Partially parsed test_simple_filter_extension_initialization. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '{{ text|reverse_string }}'
    var_1 = 'hello'

def test_case_0():
    var_0 = '{{ num|multiply(3) }}'
    var_1 = 5

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_extensions. Retrieved 9/18 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_filter'
    var_2 = bool('my_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_filter']
    var_4 = 'my_filter'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = bool('reverse_string' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_string'
    var_4 = var_0.filters[var_3]
    var_5 = 'abc'
    var_6 = var_4(var_5)
    assert var_6 == 'cba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_0.Environment()
    var_2 = 'filter1'
    var_3 = bool('filter1' in var_0.filters)
    assert var_3 is True
    var_4 = 'filter2'
    var_5 = bool('filter2' in var_1.filters)
    assert var_5 is True
    var_6 = 'filter1'
    var_7 = var_0.filters[var_6]
    var_8 = 'test'
    var_9 = var_7(var_8)
    assert var_9 == 'test1'
    var_10 = 'filter2'
    var_11 = var_1.filters[var_10]
    var_12 = var_11(var_8)
    assert var_12 == 'test2'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_extension_initialization. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_filters. Retrieved 8/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_custom_filter'
    var_2 = bool('my_custom_filter' in var_0.filters)
    assert var_2 is True
    var_3 = 'my_custom_filter'
    var_4 = var_0.filters[var_3]
    var_5 = 'hello'
    var_6 = var_4(var_5)
    assert var_6 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = bool('reverse_string' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_string'
    var_4 = var_0.filters[var_3]
    var_5 = 'abc'
    var_6 = var_4(var_5)
    assert var_6 == 'cba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_prefix'
    var_2 = bool('add_prefix' in var_0.filters)
    assert var_2 is True
    var_3 = 'add_prefix'
    var_4 = var_0.filters[var_3]
    var_5 = 'test'
    var_6 = var_4(var_5)
    assert var_6 == 'prefix_test'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'filter1'
    var_2 = var_0.filters[var_1]
    var_3 = 'test'
    var_4 = var_2(var_3)
    assert var_4 == 'test1'
    var_5 = 'filter2'
    var_6 = var_0.filters[var_5]
    var_7 = var_6(var_3)
    assert var_7 == 'test2'



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_extension_is_callable. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_multiple_arguments. Retrieved 6/11 statements.
# Partially parsed test_simple_filter_preserves_function_behavior. Retrieved 8/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_filter'
    var_2 = bool('my_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_filter']

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_numbers'
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = 3
    var_5 = var_2(var_3, var_4)
    assert var_5 == 8

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'double'
    var_2 = var_0.filters[var_1]
    var_3 = 10
    var_4 = var_2(var_3)
    assert var_4 == 20
    var_5 = var_0.filters[var_1]
    var_6 = 'ab'
    var_7 = var_5(var_6)
    assert var_7 == 'abab'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_uppercase.
# Partially parsed test_simple_filter. Retrieved 5/2 statements.
# Partially parsed test_simple_filter_multiple_filters. Retrieved 8/17 statements.
# Partially parsed test_simple_filter_with_numeric_function. Retrieved 8/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_uppercase'
    var_2 = bool('test_uppercase' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_uppercase']
    var_4 = 'test_uppercase'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_prefix'
    var_2 = bool('add_prefix' in var_0.filters)
    assert var_2 is True
    var_3 = 'add_suffix'
    var_4 = bool('add_suffix' in var_0.filters)
    assert var_4 is True
    var_5 = 'add_prefix'
    var_6 = var_0.filters[var_5]
    var_7 = 'test'
    var_8 = var_6(var_7)
    assert var_8 == 'prefix_test'
    var_9 = 'add_suffix'
    var_10 = var_0.filters[var_9]
    var_11 = var_10(var_7)
    assert var_11 == 'test_suffix'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'double'
    var_2 = bool('double' in var_0.filters)
    assert var_2 is True
    var_3 = 'double'
    var_4 = var_0.filters[var_3]
    var_5 = 5
    var_6 = var_4(var_5)
    assert var_6 == 10
    var_7 = var_0.filters[var_3]
    var_8 = 0
    var_9 = var_7(var_8)
    assert var_9 == 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_extensions. Retrieved 8/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_custom_filter'
    var_2 = bool('my_custom_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_custom_filter']
    var_4 = 'my_custom_filter'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = bool('reverse_string' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_string'
    var_4 = var_0.filters[var_3]
    var_5 = 'abc'
    var_6 = var_4(var_5)
    assert var_6 == 'cba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_exclamation'
    var_2 = bool('add_exclamation' in var_0.filters)
    assert var_2 is True
    var_3 = 'add_question'
    var_4 = bool('add_question' in var_0.filters)
    assert var_4 is True
    var_5 = 'add_exclamation'
    var_6 = var_0.filters[var_5]
    var_7 = 'hello'
    var_8 = var_6(var_7)
    assert var_8 == 'hello!'
    var_9 = 'add_question'
    var_10 = var_0.filters[var_9]
    var_11 = var_10(var_7)
    assert var_11 == 'hello?'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_filter.
# Partially parsed test_simple_filter. Retrieved 5/2 statements.
# Partially parsed test_simple_filter_with_multiple_filters. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_integration_with_template. Retrieved 1/9 statements.
# Partially parsed test_simple_filter_preserves_filter_function. Retrieved 8/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['test_filter']
    var_4 = 'test_filter'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_filter'
    var_2 = bool('reverse_filter' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_filter'
    var_4 = var_0.filters[var_3]
    var_5 = 'abc'
    var_6 = var_4(var_5)
    assert var_6 == 'cba'

def test_case_0():
    var_0 = "{{ 'x' | double_filter }}"

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = var_0.filters['custom_filter']
    var_2 = 'custom_filter'
    var_3 = var_0.filters[var_2]
    var_4 = 'test'
    var_5 = var_3(var_4)
    assert var_5 == 'test!'
    var_6 = var_0.filters[var_2]
    var_7 = '?'
    var_8 = var_6(var_4, suffix=var_7)
    assert var_8 == 'test?'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_instances. Retrieved 8/14 statements.
# Partially parsed test_simple_filter_numeric_operation. Retrieved 11/16 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_custom_filter'
    var_2 = bool('my_custom_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_custom_filter']
    var_4 = 'my_custom_filter'
    var_5 = var_0.filters[var_4]
    var_6 = 'hello'
    var_7 = var_5(var_6)
    assert var_7 == 'HELLO'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = bool('reverse_string' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_string'
    var_4 = var_0.filters[var_3]
    var_5 = 'abc'
    var_6 = var_4(var_5)
    assert var_6 == 'cba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_0.Environment()
    var_2 = 'add_prefix'
    var_3 = bool('add_prefix' in var_0.filters)
    assert var_3 is True
    var_4 = 'add_prefix'
    var_5 = bool('add_prefix' in var_1.filters)
    assert var_5 is True
    var_6 = 'add_prefix'
    var_7 = var_0.filters[var_6]
    var_8 = 'test'
    var_9 = var_7(var_8)
    assert var_9 == 'prefix_test'
    var_10 = var_1.filters[var_6]
    var_11 = var_10(var_8)
    assert var_11 == 'prefix_test'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'double_number'
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 == 10
    var_5 = var_0.filters[var_1]
    var_6 = 0
    var_7 = var_5(var_6)
    assert var_7 == 0
    var_8 = var_0.filters[var_1]
    var_9 = -3
    var_10 = var_8(var_9)
    assert var_10 == -6



