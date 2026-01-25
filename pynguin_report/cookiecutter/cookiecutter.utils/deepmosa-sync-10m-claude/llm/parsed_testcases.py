####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/6 statements.
# Failed to parse test_simple_filter_with_different_function_names.
# Partially parsed test_simple_filter_function_is_callable. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_complex_filter_function. Retrieved 5/10 statements.


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
    var_1 = 'double_value'
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 == 10

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_string'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
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
    var_1 = 'filter_one'
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 == 6
    var_5 = 'filter_two'
    var_6 = var_0.filters[var_5]
    var_7 = var_6(var_3)
    assert var_7 == 10



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_filter_works_in_template. Retrieved 4/10 statements.
# Partially parsed test_simple_filter_with_multiple_filters. Retrieved 8/17 statements.
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
    var_1 = '{{ text|reverse_string }}'
    var_2 = var_0.from_string(var_1)
    var_3 = 'hello'

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
    var_1 = 'multiply_by_two'
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 == 10
    var_5 = var_0.filters[var_1]
    var_6 = 3
    var_7 = var_5(var_6)
    assert var_7 == 6



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/12 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/11 statements.
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_simple_filter. Retrieved 2/11 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 2/10 statements.
# Failed to parse test_filter.
# Partially parsed test_simple_filter_extension_initialization. Retrieved 5/2 statements.


def test_case_0():
    var_0 = 'my_custom_filter'
    var_1 = 'my_custom_filter'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'reverse_string'
    var_1 = 'reverse_string'
    var_2 = 'hello'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_filter'
    var_2 = bool('test_filter' in var_0.filters)
    assert var_2 is True
    var_3 = 'test_filter'
    var_4 = var_0.filters[var_3]
    var_5 = 'data'
    var_6 = var_4(var_5)
    assert var_6 == 'filtered_data'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_with_different_function_names. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_function_is_callable. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_multiple_arguments. Retrieved 6/11 statements.


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
    var_1 = 'lowercase_filter'
    var_2 = bool('lowercase_filter' in var_0.filters)
    assert var_2 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'reverse_filter'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = var_2(var_3)
    assert var_4 == 'olleh'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'concat_filter'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'
    var_4 = ' world'
    var_5 = var_2(var_3, var_4)
    assert var_5 == 'hello world'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_extensions. Retrieved 8/17 statements.
# Failed to parse test_simple_filter_returns_extension_class.
# Partially parsed test_simple_filter_with_numeric_filter. Retrieved 10/15 statements.


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
    var_2 = var_0.filters[var_1]
    var_3 = 5
    var_4 = var_2(var_3)
    assert var_4 == 10
    var_5 = var_0.filters[var_1]
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = var_5(var_8)
    var_10 = bool(var_9 == [1, 2, 1, 2])
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_instances. Retrieved 8/14 statements.


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
    var_3 = var_0.filters[var_2]
    var_4 = 'test'
    var_5 = var_3(var_4)
    assert var_5 == 'prefix_test'
    var_6 = var_1.filters[var_2]
    var_7 = var_6(var_4)
    assert var_7 == 'prefix_test'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_simple_filter. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_integration_with_template. Retrieved 2/10 statements.
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
    var_1 = 'reverse_text'
    var_2 = bool('reverse_text' in var_0.filters)
    assert var_2 is True
    var_3 = 'reverse_text'
    var_4 = var_0.filters[var_3]
    var_5 = 'hello'
    var_6 = var_4(var_5)
    assert var_6 == 'olleh'

def test_case_0():
    var_0 = '{{ value|double }}'
    var_1 = 5

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/12 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_multiple_extensions. Retrieved 9/18 statements.


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
    var_5 = 'abcdef'
    var_6 = var_4(var_5)
    assert var_6 == 'fedcba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_0.Environment()
    var_2 = 'filter_one'
    var_3 = bool('filter_one' in var_0.filters)
    assert var_3 is True
    var_4 = 'filter_two'
    var_5 = bool('filter_two' in var_1.filters)
    assert var_5 is True
    var_6 = 'filter_two'
    var_7 = bool('filter_two' not in var_0.filters)
    assert var_7 is True
    var_8 = 'filter_one'
    var_9 = bool('filter_one' not in var_1.filters)
    assert var_9 is True
    var_10 = 'filter_one'
    var_11 = var_0.filters[var_10]
    var_12 = 'test'
    var_13 = var_11(var_12)
    assert var_13 == 'test_one'
    var_14 = 'filter_two'
    var_15 = var_1.filters[var_14]
    var_16 = var_15(var_12)
    assert var_16 == 'test_two'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_instances. Retrieved 8/14 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_instances. Retrieved 8/14 statements.


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
    var_1 = 'reverse_text'
    var_2 = var_0.filters[var_1]
    var_3 = 'abc'
    var_4 = var_2(var_3)
    assert var_4 == 'cba'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_0.Environment()
    var_2 = 'add_prefix'
    var_3 = var_0.filters[var_2]
    var_4 = 'test'
    var_5 = var_3(var_4)
    assert var_5 == 'prefix_test'
    var_6 = var_1.filters[var_2]
    var_7 = var_6(var_4)
    assert var_7 == 'prefix_test'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_multiple_instances. Retrieved 8/14 statements.


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



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_simple_filter_creates_extension_class.
# Failed to parse test_simple_filter_registers_filter_in_environment.
# Partially parsed test_simple_filter_extension_is_callable. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_with_string_processing. Retrieved 2/10 statements.
# Partially parsed test_simple_filter_preserves_function_behavior. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'double_value'
    var_1 = 5

def test_case_0():
    var_0 = 'add_prefix'
    var_1 = 'test'

def test_case_0():
    var_0 = 'multiply_by_three'
    var_1 = 4
    var_2 = 0
    var_3 = -2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/12 statements.
# Partially parsed test_simple_filter_with_different_function. Retrieved 5/11 statements.
# Partially parsed test_simple_filter_multiple_extensions. Retrieved 8/18 statements.


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
    var_5 = 'hello'
    var_6 = var_4(var_5)
    assert var_6 == 'olleh'

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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_simple_filter. Retrieved 5/11 statements.
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



