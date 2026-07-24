####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 1/6 statements.
# Partially parsed test_simple_filter_works_with_multiple_filters. Retrieved 1/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = var_0.filters['filter_one']
    var_2 = var_0.filters['filter_two']



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_simple_filter_adds_filter_to_environment. Retrieved 4/11 statements.
# Partially parsed test_simple_filter_sets_correct_extension_name. Retrieved 1/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'add_exclamation'
    var_2 = bool('add_exclamation' in var_0.filters)
    assert var_2 is True
    var_3 = 'add_exclamation'
    var_4 = var_0.filters[var_3]
    var_5 = 'test_val'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'mock_filter'
    var_2 = bool('mock_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['mock_filter']



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 1/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/9 statements.
# Partially parsed test_simple_filter_different_function_names. Retrieved 1/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'another_func'
    var_2 = bool('another_func' in var_0.filters)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_simple_filter_adds_filter_to_environment. Retrieved 5/10 statements.
# Partially parsed test_simple_filter_preserves_function_name_in_extension. Retrieved 1/5 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_custom_filter'
    var_2 = bool('my_custom_filter' in var_0.filters)
    assert var_2 is True
    var_3 = 'my_custom_filter'
    var_4 = var_0.filters[var_3]
    var_5 = 123
    var_6 = var_4(var_5)
    assert var_6 == '123'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_simple_filter_registers_filter_in_environment.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 4/29 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'jinja2'
    var_2 = lambda x: x
    var_3 = module_0.Environment()
    var_4 = 'my_filter'
    var_5 = bool('my_filter' in var_3.filters)
    assert var_5 is True
    var_6 = var_3.filters['my_filter']
    var_7 = bool(var_3.filters['my_filter'] == var_2)
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_sure_path_exists_creates_nested_directories. Retrieved 8/12 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_dir_new'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1.rmtree(var_3)
    var_5 = module_1.make_sure_path_exists(var_3)
    var_6 = var_3.is_dir()
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.rmtree(var_3)

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_dir_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = module_1.make_sure_path_exists(var_3)
    var_7 = var_3.is_dir()
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = module_1.rmtree(var_3)

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test/nested/deep/dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test'
    var_5 = module_1.rmtree(var_4)
    var_6 = module_1.make_sure_path_exists(var_3)
    var_7 = var_3.is_dir()
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = 'test'
    var_10 = module_1.rmtree(var_9)

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_file_blocking'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'test_file_blocking/sub_dir'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1.make_sure_path_exists(var_8)
    var_10 = var_3.unlink()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 4/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'mocklan_filter'
    var_2 = locals()
    var_3 = var_1 in var_2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_simple_filter_registers_filter_in_environment. Retrieved 1/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'my_custom_filter'
    var_2 = bool('my_custom_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['my_custom_filter']



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 1/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_simple_filter_adds_filter_to_environment. Retrieved 1/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'mock_filter'
    var_2 = bool('mock_filter' in var_0.filters)
    assert var_2 is True
    var_3 = var_0.filters['mock_filter']



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_simple_filter_adds_filter_to_environment. Retrieved 1/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = var_0.filters['mock_filter']



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment. Retrieved 1/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_simple_filter_adds_to_environment_filters. Retrieved 1/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_simple_filter_registers_function_in_environment_filters. Retrieved 6/28 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_0.Environment()
    var_2 = 'my_test_filter'
    var_3 = bool('my_test_filter' in var_1.filters)
    assert var_3 is True
    var_4 = 'my_test_filter'
    var_5 = var_1.filters[var_4]
    var_6 = 10
    var_7 = var_5(var_6)
    assert var_7 == 10



