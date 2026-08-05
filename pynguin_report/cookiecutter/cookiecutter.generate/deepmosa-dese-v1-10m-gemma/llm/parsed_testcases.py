####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_calls_correct_function_and_warns. Retrieved 10/21 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = '/tmp/repo'
    var_2 = 'post_gen_project'
    var_3 = '/tmp/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = 0



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'old'
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'city'
    var_6 = 'new'
    var_7 = 'London'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = 'medium'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'ultra'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-bool'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'font'
    var_3 = 'dark'
    var_4 = 'serif'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'light'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'options'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'c'
    var_8 = [var_7]
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_6, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_4, var_3: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'unrelated'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'debug'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = 'cache'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'db'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_2]
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'env'
    var_1 = 'dev'
    var_2 = 'prod'
    var_3 = 'staging'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'env'
    var_1 = 'dev'
    var_2 = 'prod'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'staging'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'db'
    var_2 = 'host'
    var_3 = 'port'
    var_4 = 'localhost'
    var_5 = 5432
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'remote'
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_8, var_12)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 5/18 statements.
# Partially parsed test_render_and_create_dir_raises_error_if_exists_and_no_overwrite. Retrieved 5/17 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 5/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = '{{cookiecutter.name}}'

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.name}}'
    var_4 = False

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.name}}'
    var_4 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "old_name", "version": "1.0.0"}'
    var_1 = 'project_name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = 'new_var'
    var_5 = 'extra_name'
    var_6 = 'new_val'
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = 'cookiecutter.json'
    var_9 = module_0.generate_context(var_8, var_3, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'overwritten_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "missing_quote}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'test.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true_at_line_38. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overridden_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_apply_overwrites_to_context_invalid_boolean_conversion_raises_value_error. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean-value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 6/12 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 11/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 6/11 statements.
# Partially parsed test_generate_file_empty_output_path. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'src/data.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/output/data.bin'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'src/main.py'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'my_project'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'src/main.py'
    var_10 = "print('hello')"

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'src/config.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = True

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'src/{template_name}/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'template_name'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'config/*.json'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'config/settings.json'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'test.py'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False
    var_8 = 'config/settings.json'
    var_9 = module_0.is_copy_only_path(var_8, var_5)
    assert var_9 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'other'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_success_path. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'test_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_default_context_is_none. Retrieved 8/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 38 evaluates to False by passing None as default_context.'
    var_1 = 'test_context.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = None
    var_7 = module_0.generate_context(var_1, var_6)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 18/32 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_underlying_function. Retrieved 8/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'repo'
    var_2 = 'post_gen_project'
    var_3 = 'project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = -1
    var_10 = -1
    var_11 = 'repo'
    var_12 = 'post_gen_project'
    var_13 = 'project'
    var_14 = 'foo'
    var_15 = 'bar'
    var_16 = {var_14: var_15}
    var_17 = True

def test_case_0():
    var_0 = '/repo'
    var_1 = 'pre'
    var_2 = '/proj'
    var_3 = 'ctx'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = (var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = 'logging'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'db'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'env'
    var_1 = 'dev'
    var_2 = 'prod'
    var_3 = 'staging'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'env'
    var_1 = 'dev'
    var_2 = 'prod'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'staging'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = False
    var_4 = 8080
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'sub'
    var_1 = 'existing'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tags'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_exists_without_overwrite_raises_exception. Retrieved 7/15 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite_success. Retrieved 9/18 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/test'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}_dir'
    var_5 = 'output'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'existing'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'
    var_5 = 'output'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'overwrite_me'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'
    var_5 = 'output'
    var_6 = 'old_file.txt'
    var_7 = 'old'
    var_8 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 8/18 statements.
# Partially parsed test_render_and_create_dir_raises_if_exists_without_overwrite. Retrieved 8/17 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 9/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = './test_dir_new'
    var_5 = '{{cookiecutter.name}}'
    var_6 = './'
    var_7 = './my_project'

def test_case_0():
    var_0 = 'name'
    var_1 = 'existing_dir'
    var_2 = {var_0: var_1}
    var_3 = './existing_dir'
    var_4 = True
    var_5 = '{{cookiecutter.name}}'
    var_6 = './'
    var_7 = False

def test_case_0():
    var_0 = 'name'
    var_1 = 'overwrite_dir'
    var_2 = {var_0: var_1}
    var_3 = './overwrite_dir'
    var_4 = True
    var_5 = '{{cookiecutter.name}}'
    var_6 = './'
    var_7 = True
    var_8 = './overwrite_dir'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_error. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 8/20 statements.
# Partially parsed test_render_and_create_dir_already_exists_raises_error. Retrieved 7/13 statements.
# Partially parsed test_render_and_create_dir_overwrite_success. Retrieved 8/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = False
    var_5 = '{{cookiecutter.name}}'
    var_6 = '/tmp/out'
    var_7 = '/tmp/out/my_project'

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = '{{cookiecutter.name}}'
    var_5 = '/tmp/out'
    var_6 = False

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = '{{cookiecutter.name}}'
    var_5 = '/tmp/out'
    var_6 = True
    var_7 = '/tmp/out/my_project'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.
# Partially parsed test_generate_context_preserves_order. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'overwritten_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'ContextDecodingException not raised'
    var_4 = AssertionError(var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'cookiecutter'
    var_4 = var_2[var_3]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_path_already_exists. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'test_output_dir'
    var_1 = 'template_name'
    var_2 = {}
    var_3 = 'rendered_name'
    var_4 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_renders_name_correctly. Retrieved 5/10 statements.
# Partially parsed test_render_and_create_dir_success. Retrieved 5/10 statements.
# Partially parsed test_render_and_create_dir_raises_output_dir_exists_exception. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_overwrites_if_enabled. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'project_{{ name }}'
    var_1 = 'name'
    var_2 = 'my_app'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'

def test_case_0():
    var_0 = 'project_{{ name }}'
    var_1 = 'name'
    var_2 = 'my_app'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test'
    var_2 = ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 7/17 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'test_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 6/14 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'config/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'config/settings.json'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'script.py'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'any_file.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'test.txt'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'README.md'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.is_copy_only_path(var_2, var_5)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 7/16 statements.
# Partially parsed test_render_and_create_dir_already_exists_raises_exception. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_overwrite_allowed. Retrieved 5/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = '/tmp/test'

def test_case_0():
    var_0 = 'name'
    var_1 = 'world'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = 'rendered_name'
    var_5 = '{{ name }}'
    var_6 = False

def test_case_0():
    var_0 = 'output'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '{{ name }}'
    var_4 = False

def test_case_0():
    var_0 = 'output'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '{{ name }}'
    var_4 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_enters_overwrite_logic. Retrieved 10/29 statements.


def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = 'project_name'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = 'project_{{ name }}'
    var_7 = 'test'
    var_8 = {var_2: var_7}
    var_9 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/11 statements.
# Partially parsed test_generate_file_text_rendering_with_custom_newlines. Retrieved 13/24 statements.
# Partially parsed test_generate_file_skips_if_exists. Retrieved 6/12 statements.
# Partially parsed test_generate_file_empty_name_returns_early. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/binary.dat'
    var_2 = {}
    var_3 = '/tmp/output/binary.dat'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/config.j2'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = 'test'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'Hello {{ name }}'
    var_10 = 'Hello test'
    var_11 = 0
    var_12 = 1

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/existing.txt'
    var_2 = {}
    var_3 = True
    var_4 = 'The resulting file already exists: %s'
    var_5 = '/tmp/output/existing.txt'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/dir_as_file.j2'
    var_2 = {}
    var_3 = 'The resulting file name is empty: %s'
    var_4 = '/tmp/output/dir_as_file'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_raises_error_if_exists_and_no_overwrite. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_success_overwrite_existing. Retrieved 5/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test_out'
    var_2 = ''

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/test_out'
    var_5 = 'my_project'
    var_6 = '/tmp/test_out/my_project'

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/test_out'
    var_3 = False

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/test_out'
    var_3 = True
    var_4 = '/tmp/test_out/existing_dir'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_with_order_preservation. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "test_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = 'new_var'
    var_7 = '2.0.0'
    var_8 = 'new_val'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.generate_context(var_1, var_4, var_9)
    var_11 = {var_2: var_3, var_5: var_7}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "test_project", }'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'ContextDecodingException not raised'
    var_4 = AssertionError(var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 'config.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'config'
    var_4 = var_2[var_3]
    var_5 = var_2[var_3]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_file_template_syntax_error_exception_translation_is_false. Retrieved 9/20 statements.


import jinja2.exceptions as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'Syntax error'
    var_6 = 'template.txt'
    var_7 = 1
    var_8 = module_0.TemplateSyntaxError(var_6, var_7)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'unrelated'
    var_4 = 'new'
    var_5 = 'value'
    var_6 = {var_0: var_4, var_3: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_2, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'font'
    var_6 = 'roboto'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'z'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'first'
    var_2 = 'second'
    var_3 = 'third'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = 2
    var_9 = {var_7: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'tags'
    var_2 = 'old'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'new'
    var_7 = [var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_5, var_9, in_dictionary_variable=var_10)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/12 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 12/24 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 6/13 statements.
# Partially parsed test_generate_file_empty_name_directory. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/bin.dat'
    var_2 = {}
    var_3 = 'bin.dat'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'val'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'Hello {{ var }}'
    var_10 = 'Hello val'
    var_11 = 'template/file.txt'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/exists.txt'
    var_2 = {}
    var_3 = True
    var_4 = 'The resulting file already exists: %s'
    var_5 = 'exists.txt'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/dir_tmpl'
    var_2 = {}
    var_3 = 'The resulting file name is empty: %s'
    var_4 = 'dir_tmpl'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_invalid_raises_value_error. Retrieved 6/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 7/14 statements.
# Partially parsed test_generate_context_with_overwrites. Retrieved 12/18 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 3/11 statements.
# Partially parsed test_generate_context_warning_on_invalid_default. Retrieved 14/25 statements.


import json as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)

import json as module_0

def test_case_0():
    var_0 = 'config.json'
    var_1 = 'project_name'
    var_2 = 'debug'
    var_3 = 'original'
    var_4 = False
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = 'default'
    var_8 = {var_1: var_7}
    var_9 = 'extra'
    var_10 = 'true'
    var_11 = {var_1: var_9, var_2: var_10}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bad.json'
    var_1 = '{ invalid json }'
    var_2 = module_0.generate_context(var_0)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'choice'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    assert var_4 == 1
    var_5 = {var_1: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = 'c'
    var_8 = [var_7]
    var_9 = {var_1: var_8}
    var_10 = 'always'
    var_11 = module_1.generate_context(var_2, var_9)
    var_12 = 0
    var_13 = str(var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_predicate_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.txt'
    var_8 = lambda **kwargs: var_7
    var_9 = 'rendered content'
    var_10 = lambda **kwargs: var_9
    var_11 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_file_is_binary_true. Retrieved 7/14 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'binary_file.bin'
    var_6 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_binary_path_evaluates_true_at_line_47. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test_binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_binary.bin'
    var_6 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_generate_file_is_binary_true. Retrieved 9/21 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'test_binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_binary_file.bin'
    var_6 = True
    var_7 = b'\x00\x01\x02\x03'
    var_8 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new'
    var_7 = 'admin'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'font'
    var_6 = 'serif'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'metadata'
    var_2 = 'tags'
    var_3 = 'python'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_0, var_6, in_dictionary_variable=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'z'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = 'medium'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'turbo'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'x'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'db'
    var_2 = 'user'
    var_3 = 'port'
    var_4 = 'root'
    var_5 = 5432
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'host'
    var_10 = 9999
    var_11 = 'localhost'
    var_12 = {var_3: var_10, var_9: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_0: var_13}
    var_15 = module_0.apply_overwrites_to_context(var_8, var_14)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 12/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = 'new_var'
    var_7 = '2.0.0'
    var_8 = 'new_val'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.generate_context(var_1, var_4, var_9)
    var_11 = {var_2: var_3, var_5: var_7, var_6: var_8}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project",'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'ContextDecodingException not raised'
    var_4 = AssertionError(var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"list_var": ["a", "b"], "dict_var": {"key": "old"}}'
    var_1 = 'cookiecutter.json'
    var_2 = 'list_var'
    var_3 = 'dict_var'
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = 'added'
    var_8 = 'new'
    var_9 = 'true'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_2: var_5, var_3: var_10}
    var_12 = module_0.generate_context(var_1, extra_context=var_11)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite_raises_exception. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite_success. Retrieved 5/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'
    var_5 = 'my_project'

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = False

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = True
    var_4 = 'project'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 18/32 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_correct_function. Retrieved 13/18 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'repo'
    var_2 = 'post_gen_project'
    var_3 = 'project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = -1
    var_10 = -1
    var_11 = 'repo'
    var_12 = 'post_gen_project'
    var_13 = 'project'
    var_14 = 'foo'
    var_15 = 'bar'
    var_16 = {var_14: var_15}
    var_17 = True

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'context'
    var_4 = 'delete_project_on_failure'
    var_5 = '/tmp/repo'
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = False
    var_12 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_10, var_4: var_11}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test_dir'
    var_2 = ''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_success_new_path. Retrieved 7/21 statements.
# Partially parsed test_render_and_create_dir_already_exists_no_overwrite. Retrieved 6/17 statements.
# Partially parsed test_render_and_create_dir_already_exists_with_overwrite. Retrieved 5/15 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/out'

def test_case_0():
    var_0 = '/tmp/cookiecutter_test_dir'
    var_1 = True
    var_2 = 'name'
    var_3 = 'world'
    var_4 = {var_2: var_3}
    var_5 = '{{ cookiecutter.name }}'
    var_6 = '/tmp/cookiecutter_test_dir/rendered_name'

def test_case_0():
    var_0 = '/tmp/cookiecutter_exists_test'
    var_1 = 'existing_dir'
    var_2 = True
    var_3 = '{{ cookiecutter.name }}'
    var_4 = {}
    var_5 = False

def test_case_0():
    var_0 = '/tmp/cookiecutter_overwrite_test'
    var_1 = 'overwrite_test'
    var_2 = True
    var_3 = '{{ cookiecutter.name }}'
    var_4 = {}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}

def test_case_0():
    var_0 = '{"key": "missing_quote}'
    var_1 = 'cookiecutter.json'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_context_with_default_context_triggers_predicate. Retrieved 8/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Ensure that the predicate 'if default_context:' evaluates to True."
    var_1 = 'test_context.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'overridden_project'
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_1, var_6)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_with_default_context_triggers_predicate. Retrieved 8/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that the predicate 'if default_context:' evaluates to True."
    var_1 = 'test_cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'overridden_name'
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_1, var_6)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'config/'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'notes.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'config/settings.json'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'script.py'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'notes.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'notes.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'notes.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 14/37 statements.
# Partially parsed test_generate_files_empty_context_error. Retrieved 9/20 statements.


def test_case_0():
    var_0 = './test_repo'
    var_1 = './test_output'
    var_2 = 'cookiecutter-test_{{ project_slug }}'
    var_3 = True
    var_4 = 'files'
    var_5 = 'hello.txt'
    var_6 = 'Hello {{ project_slug }}!'
    var_7 = 'cookiecutter'
    var_8 = '_jinja2_env_vars'
    var_9 = 'project_slug'
    var_10 = 'my_project'
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_7: var_11, var_8: var_12}

import cookiecutter.utils as module_0

def test_case_0():
    var_0 = './test_repo_fail'
    var_1 = 'cookiecutter-test_stub'
    var_2 = True
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = './test_output_fail'
    var_7 = './test_output_fail'
    var_8 = module_0.rmtree(var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = 'static/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'data.bin'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'static/logo.png'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'src/main.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'src/main.py'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'any_path'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False



# Parsed testcases at query #13
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = False
    var_4 = 8080
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_key'
    var_8 = True
    var_9 = 'val'
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = 'red'
    var_2 = 'blue'
    var_3 = 'green'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = 'red'
    var_2 = 'blue'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'green'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = 'cache'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'cache'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'cache'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-bool'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = 2
    var_9 = 3
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_6, var_12, in_dictionary_variable=var_13)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 4/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''
    var_3 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_successful_creation. Retrieved 5/18 statements.
# Partially parsed test_render_and_create_dir_already_exists_raises_exception. Retrieved 5/19 statements.
# Partially parsed test_render_and_create_dir_overwrite_enabled. Retrieved 5/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'my-project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = '{{cookiecutter.name}}'

def test_case_0():
    var_0 = 'name'
    var_1 = 'my-project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.name}}'
    var_4 = False

def test_case_0():
    var_0 = 'name'
    var_1 = 'my-project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.name}}'
    var_4 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_files_skips_copy_only_directories_in_walk. Retrieved 16/31 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/repo/template_dir'
    var_1 = '/output/project'
    var_2 = True
    var_3 = '.'
    var_4 = 'render_me'
    var_5 = 'copy_me'
    var_6 = [var_4, var_5]
    var_7 = 'file1.txt'
    var_8 = [var_7]
    var_9 = (var_3, var_6, var_8)
    var_10 = '/repo'
    var_11 = 'cookiecutter'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = '/output'
    var_15 = module_0.generate_files(var_10, var_13, var_14)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_context_predicate_false_when_no_default_context. Retrieved 6/14 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'overridden_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "incomplete"'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'new_var'
    var_3 = 'should_not_appear'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'extra'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, extra_context=var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_delete_project_on_failure_is_true. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = '/tmp/repo'
    var_3 = '/tmp/repo'
    var_4 = {}
    var_5 = False
    var_6 = module_0.generate_files(var_3, var_4, keep_project_on_failure=var_5)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_failure. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_context_predicate_false_when_no_default_context. Retrieved 6/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_files_basic_rendering. Retrieved 12/36 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 14/36 statements.


def test_case_0():
    var_0 = 'cookiecutter-test-template'
    var_1 = 'cookiecutter.project_name'
    var_2 = '_jinja2_env_vars'
    var_3 = 'My Project'
    var_4 = {}
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Project Name: {{ cookiecutter.project_name }}'
    var_7 = 'README.md'
    var_8 = '{{ cookiecutter.project_name }}_dir'
    var_9 = 'cookiecutter'
    var_10 = {var_9: var_5}
    var_11 = 'My Project_dir'

def test_case_0():
    var_0 = 'cookiecutter-test-template'
    var_1 = 'cookiecutter.project_name'
    var_2 = 'cookiecutter'
    var_3 = 'My Project'
    var_4 = '_copy_without_render'
    var_5 = '*.txt'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_1: var_3, var_2: var_7}
    var_9 = 'This {{ variable }} should remain unchanged.'
    var_10 = 'ignore_me.txt'
    var_11 = 'hello.txt'
    var_12 = 'Hello {{ cookiecutter.project_name }}'
    var_13 = {var_2: var_8}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_files_delete_project_on_failure_is_false_when_output_directory_not_created. Retrieved 7/14 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = False
    var_2 = '/tmp/template'
    var_3 = {}
    var_4 = '/tmp/output'
    var_5 = module_0.generate_files(var_2, var_3, var_4)
    var_6 = 4



