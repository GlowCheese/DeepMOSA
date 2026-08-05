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
    var_7 = 'dev'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'font'
    var_3 = 'dark'
    var_4 = 'serif'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'size'
    var_8 = 'light'
    var_9 = 12
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_12)
    var_14 = var_6['settings']['theme']
    assert var_14 == 'light'
    var_15 = var_6['settings']['font']
    assert var_15 == 'serif'
    var_16 = var_6['settings']['size']
    assert var_16 == 12

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'languages'
    var_1 = 'python'
    var_2 = 'javascript'
    var_3 = 'rust'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['languages']
    var_10 = bool(var_5['languages'] == ['python', 'rust'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'languages'
    var_1 = 'python'
    var_2 = 'javascript'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'cpp'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = 'strawberry'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['flavor'][0]
    assert var_8 == 'chocolate'
    var_9 = 'vanilla'
    var_10 = bool('vanilla' in var_5['flavor'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'strawberry'
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
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['items']
    var_12 = bool(var_4['items'] == ['c', 'd'])
    assert var_12 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_list'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.apply_overwrites_to_context(var_5, var_7, in_dictionary_variable=var_8)
    var_10 = var_5['my_list']
    var_11 = bool(var_5['my_list'] == ['a', 'b'])
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 6/15 statements.
# Partially parsed test_render_and_create_dir_already_exists_raises_exception. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_overwrite_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = '/tmp'

def test_case_0():
    var_0 = 'name'
    var_1 = 'world'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = '{{cookiecutter.name}}'
    var_5 = 'rendered_name'

def test_case_0():
    var_0 = 'output'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '{{cookiecutter.name}}'
    var_4 = False

def test_case_0():
    var_0 = 'output'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '{{cookiecutter.name}}'
    var_4 = True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'nested_key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_value'
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = var_4['my_var']['nested_key']
    assert var_10 == 'new_value'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_logic. Retrieved 9/13 statements.
# Partially parsed test_run_hook_from_repo_dir_deprecation_warning_details. Retrieved 6/9 statements.


def test_case_0():
    pass

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'post_gen_project'
    var_2 = 'project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'r'
    var_1 = 'h'
    var_2 = 'p'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = "The '_run_hook_from_repo_dir' function is deprecated"



# Parsed testcases at query #6
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
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'debug'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'port'
    var_6 = 8080
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['config']['port']
    assert var_11 == 8080
    var_12 = var_4['config']['debug']
    assert var_12 is False

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
    var_9 = var_5['features']
    var_10 = bool(var_5['features'] == ['auth', 'cache'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'database'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'env'
    var_1 = 'dev'
    var_2 = 'prod'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = var_4['env'][0]
    assert var_7 == 'prod'
    var_8 = var_4['env'][1]
    assert var_8 == 'dev'

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
    var_0 = 'tags'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_1, var_2]
    var_10 = {var_0: var_9}
    var_11 = [var_5, var_6]
    var_12 = {var_0: var_11}
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_10, var_12, in_dictionary_variable=var_13)
    var_15 = 'inner'
    var_16 = [var_1, var_2]
    var_17 = {var_0: var_16}
    var_18 = {var_15: var_17}
    var_19 = [var_5, var_6]
    var_20 = {var_0: var_19}
    var_21 = {var_15: var_20}
    var_22 = module_0.apply_overwrites_to_context(var_18, var_21, in_dictionary_variable=var_13)
    var_23 = var_18['inner']['tags']
    var_24 = bool(var_18['inner']['tags'] == ['c', 'd'])
    assert var_24 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'not-a-bool'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'size'
    var_3 = 'light'
    var_4 = 12
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'dark'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = var_6['settings']['theme']
    assert var_11 == 'dark'
    var_12 = var_6['settings']['size']
    assert var_12 == 12



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_success_new_path. Retrieved 6/15 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 5/16 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 6/18 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/out'
    var_3 = ''
    var_4 = {}
    var_5 = '/tmp/out'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'name'
    var_5 = 'project_name'
    var_6 = {var_4: var_5}
    var_7 = '{{cookiecutter.name}}'
    var_8 = bool(var_4)
    assert var_8 is True

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = '{{cookiecutter.name}}'
    var_4 = False

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'dummy.txt'
    var_2 = 'name'
    var_3 = {var_2: var_0}
    var_4 = '{{cookiecutter.name}}'
    var_5 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname_string. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter-test'
    var_2 = ''
    var_3 = str(var_2)
    assert var_3 == 'Error: directory name is empty'

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter-test'
    var_2 = ''



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['is_enabled']
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True
    var_5 = var_2['cookiecutter']['project_name']
    assert var_5 == 'my_project'
    var_6 = var_2['cookiecutter']['version']
    assert var_6 == '1.0.0'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "old_name", "version": "1.0.0"}'
    var_1 = 'project_name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = 'author'
    var_5 = 'new_name'
    var_6 = 'tester'
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = 'cookiecutter.json'
    var_9 = module_0.generate_context(var_8, var_3, var_7)
    var_10 = var_9['cookiecutter']['project_name']
    assert var_10 == 'new_name'
    var_11 = var_9['cookiecutter']['version']
    assert var_11 == '1.0.0'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "missing_quote}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'JSON decoding error'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_context_successful_load. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = 'test_cookiecutter'
    var_9 = bool('test_cookiecutter' in var_6)
    assert var_9 is True
    var_10 = var_6['test_cookiecutter']['project_name']
    assert var_10 == 'my_project'
    var_11 = var_6['test_cookiecutter']['version']
    assert var_11 == '1.0.0'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_success_path. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '0.1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = 'test_context'
    var_9 = bool('test_context' in var_6)
    assert var_9 is True
    var_10 = var_6['test_context']['project_name']
    assert var_10 == 'my_project'
    var_11 = var_6['test_context']['version']
    assert var_11 == '0.1.0'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_config.json'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = bool(var_1)
    assert var_5 is True
    var_6 = var_4['test_config']
    var_7 = bool(var_4['test_config'] == var_3)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'config/settings.json'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = module_0.is_copy_only_path(var_3, var_6)
    assert var_9 is True

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
    var_8 = 'config/settings.json'
    var_9 = module_0.is_copy_only_path(var_8, var_5)
    assert var_9 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

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
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'font'
    var_3 = 'light'
    var_4 = 'Arial'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'size'
    var_8 = 'dark'
    var_9 = 12
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_12)
    var_14 = var_6['settings']['theme']
    assert var_14 == 'dark'
    var_15 = var_6['settings']['font']
    assert var_15 == 'Arial'
    var_16 = var_6['settings']['size']
    assert var_16 == 12

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
    var_9 = var_5['options']
    var_10 = bool(var_5['options'] == ['a', 'c'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
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
    var_3 = 'auto'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['mode']
    var_9 = bool(var_5['mode'] == ['slow', 'fast', 'auto'])
    assert var_9 is True

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
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_13 = module_0.apply_overwrites_to_context(var_6, var_12)
    var_14 = var_6['a']['b']['c']
    assert var_14 == 2
    var_15 = var_6['a']['b']['d']
    assert var_15 == 3

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'old'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'new'
    var_5 = 'added'
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_3, var_7, in_dictionary_variable=var_8)
    var_10 = var_3['items']
    var_11 = bool(var_3['items'] == ['new', 'added'])
    assert var_11 is True



# Parsed testcases at query #2
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
    var_7 = 'dev'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)
    var_9 = var_2['config']
    var_10 = bool(var_2['config'] == {'key': 'value'})
    assert var_10 is True

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
    var_9 = var_5['options']
    var_10 = bool(var_5['options'] == ['a', 'c'])
    assert var_10 is True

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
    var_1 = 'apple'
    var_2 = 'banana'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = var_4['choice']
    var_8 = bool(var_4['choice'] == ['banana', 'apple'])
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'apple'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'banana'
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'debug'
    var_3 = 'dark'
    var_4 = False
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'port'
    var_8 = 'light'
    var_9 = 8080
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['settings']['theme']
    assert var_13 == 'light'
    var_14 = var_6['settings']['debug']
    assert var_14 is False
    var_15 = var_6['settings']['port']
    assert var_15 == 8080

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = var_4['items']
    var_11 = bool(var_4['items'] == ['c'])
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'description'
    var_6 = 'new'
    var_7 = 'added'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = var_4['description']
    assert var_12 == 'added'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = 'new_var'
    var_8 = bool('new_var' not in var_2)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'font'
    var_6 = 'arial'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['settings']['font']
    assert var_11 == 'arial'
    var_12 = var_4['settings']['theme']
    assert var_12 == 'dark'

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
    var_9 = var_5['options']
    var_10 = bool(var_5['options'] == ['a', 'c'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
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
    var_0 = 'choice'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = 'three'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice'][0]
    assert var_8 == 'two'
    var_9 = var_5['choice']
    var_10 = bool(var_5['choice'] == ['two', 'one', 'three'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'four'
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
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_0 = 'data'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'c'
    var_8 = 'd'
    var_9 = [var_7, var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_12)
    var_14 = var_6['data']['items']
    var_15 = bool(var_6['data']['items'] == ['c', 'd'])
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'inner'
    var_2 = 'key'
    var_3 = 'old'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'added'
    var_8 = 'new'
    var_9 = True
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_6, var_12)
    var_14 = var_6['nested']['inner']['key']
    assert var_14 == 'new'
    var_15 = var_6['nested']['inner']['added']
    assert var_15 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_raises_error_if_exists_and_no_overwrite. Retrieved 8/14 statements.
# Partially parsed test_render_and_create_dir_overwrites_when_flag_is_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = '/tmp/test'

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter_test'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{ cookiecutter_name }}'
    var_8 = False
    var_9 = 'project_name'
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)

import pathlib as module_0

def test_case_0():
    var_0 = 'existing_dir_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = {}
    var_7 = '.'
    var_8 = '{{ name }}'
    var_9 = False

import pathlib as module_0

def test_case_0():
    var_0 = 'overwrite_dir_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = {}
    var_7 = '.'
    var_8 = '{{ name }}'
    var_9 = 'overwrite_test'
    var_10 = [var_7, var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 29/44 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = '_new_lines'
    var_4 = '*.txt'
    var_5 = [var_4]
    var_6 = '\n'
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 'test_project'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = '{{'
    var_11 = ''
    var_12 = '}}'
    var_13 = '/output/test_project'
    var_14 = True
    var_15 = '.'
    var_16 = 'subdir'
    var_17 = [var_16]
    var_18 = 'file1.txt'
    var_19 = 'file2.j2'
    var_20 = [var_18, var_19]
    var_21 = (var_15, var_17, var_20)
    var_22 = [var_16]
    var_23 = [var_18, var_19]
    var_24 = (var_15, var_22, var_23)
    var_25 = '/repo'
    var_26 = '/output'
    var_27 = module_0.generate_files(var_25, var_9, var_26, var_14, accept_hooks=var_14)
    assert var_27 == '/output/test_project'
    var_28 = 'pre_gen_project'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"invalid_json":'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'JSON decoding error'
    var_4 = 'ContextDecodingException not raised'
    var_5 = AssertionError(var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['cookiecutter']['project_name']
    assert var_3 == 'original'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'custom_config.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'custom_config'
    var_4 = bool('custom_config' in var_2)
    assert var_4 is True
    var_5 = var_2['custom_config']['key']
    assert var_5 == 'value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 4/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''
    var_3 = str(var_2)
    assert var_3 == 'Error: directory name is empty'



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean-choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_context_opens_file_successfully. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_cookiecutter.json'
    var_4 = module_0.generate_context(var_3)
    var_5 = 'test_cookiecutter'
    var_6 = bool('test_cookiecutter' in var_4)
    assert var_6 is True
    var_7 = var_4['test_cookiecutter']['project_name']
    assert var_7 == 'test_project'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 14/28 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_underlying_function. Retrieved 9/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'repo'
    var_2 = 'post_gen_project'
    var_3 = 'project'
    var_4 = {}
    var_5 = True
    var_6 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)
    var_7 = -1
    var_8 = -1
    var_9 = "The '_run_hook_from_repo_dir' function is deprecated"
    var_10 = 'repo'
    var_11 = 'post_gen_project'
    var_12 = 'project'
    var_13 = {}
    var_14 = True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}



# Parsed testcases at query #11
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
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

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
    var_11 = var_4['settings']['font']
    assert var_11 == 'roboto'
    var_12 = var_4['settings']['theme']
    assert var_12 == 'dark'

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
    var_9 = var_5['options']
    var_10 = bool(var_5['options'] == ['a', 'c'])
    assert var_10 is True

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
    var_9 = 'but valid choices are'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice'][0]
    assert var_8 == 'b'
    var_9 = var_5['choice']
    var_10 = bool(var_5['choice'] == ['b', 'a', 'c'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'z'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'but the choices are'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'db'
    var_1 = 'host'
    var_2 = 'port'
    var_3 = 'localhost'
    var_4 = 5432
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'user'
    var_8 = 9999
    var_9 = 'postgres'
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['db']['host']
    assert var_13 == 'localhost'
    var_14 = var_6['db']['port']
    assert var_14 == 9999
    var_15 = var_6['db']['user']
    assert var_15 == 'postgres'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tags'
    var_1 = 'old'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'new'
    var_5 = 'latest'
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_3, var_7, in_dictionary_variable=var_8)
    var_10 = var_3['tags']
    var_11 = bool(var_3['tags'] == ['new', 'latest'])
    assert var_11 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_skips_overwrite_logic_when_dir_does_not_exist. Retrieved 6/21 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp/cookiecutter_test'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = False
    var_7 = 'output'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project"}'
    var_1 = 'config.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['config']
    var_4 = bool(var_2['config'] == {'project_name': 'my_project'})
    assert var_4 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "missing_quote}'
    var_1 = 'bad.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'JSON decoding error'
    var_4 = 'ContextDecodingException not raised'
    var_5 = AssertionError(var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"settings": {"theme": "light", "debug": false}, "tags": ["web"]}'
    var_1 = 'test.json'
    var_2 = 'settings'
    var_3 = 'tags'
    var_4 = 'theme'
    var_5 = 'dark'
    var_6 = {var_4: var_5}
    var_7 = 'web'
    var_8 = 'api'
    var_9 = [var_7, var_8]
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.generate_context(var_1, extra_context=var_10)
    var_12 = var_11['test']['settings']['theme']
    assert var_12 == 'dark'
    var_13 = var_11['test']['settings']['debug']
    assert var_13 is False
    var_14 = var_11['test']['tags']
    var_15 = bool(var_11['test']['tags'] == ['web', 'api'])
    assert var_15 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_and_create_dir_skips_creation_when_dir_exists_and_overwrite_is_true. Retrieved 6/15 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_project'
    var_5 = {}
    var_6 = var_3 / var_4
    var_7 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 8/11 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite_raises_exception. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite_returns_existing. Retrieved 9/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = '/tmp/cookiecutter/my_project'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = False

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = True
    var_9 = '/tmp/cookiecutter/my_project'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'config/settings.json'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'src/assets/*'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'src/assets/logo.png'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.py'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'README.md'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'any/path'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'some/path'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'some/path'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/8 statements.
# Partially parsed test_generate_context_handles_invalid_json_by_raising_exception. Retrieved 3/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'test_context'
    var_6 = bool('test_context' in var_4)
    assert var_6 is True
    var_7 = var_4['test_context']
    var_8 = bool(var_4['test_context'] == var_3)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid_context.json'
    var_1 = '{ invalid json content '
    var_2 = module_0.generate_context(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_invalid. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean-value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_successfully_reads_file. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = 'test_context'
    var_9 = bool('test_context' in var_6)
    assert var_9 is True
    var_10 = var_6['test_context']['project_name']
    assert var_10 == 'my_project'
    var_11 = var_6['test_context']['version']
    assert var_11 == '1.0.0'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/output'
    var_2 = ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_skips_creation_when_dir_exists_and_overwrite_is_true. Retrieved 7/16 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_project'
    var_5 = {}
    var_6 = True
    var_7 = var_3.mkdir(parents=var_6, exist_ok=var_6)
    var_8 = var_3 / var_4



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_success_overwrite. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_error_already_exists. Retrieved 6/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = '{{cookiecutter.name}}'

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = '{{cookiecutter.name}}'
    var_5 = True

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = '{{cookiecutter.name}}'
    var_5 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/12 statements.
# Partially parsed test_generate_context_with_warnings_on_invalid_default. Retrieved 8/16 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "missing_quote}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['cookiecutter']['project_name']
    assert var_3 == 'original'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"choice": ["a", "b"]}'
    var_1 = 'cookiecutter.json'
    var_2 = 'choice'
    var_3 = 'c'
    var_4 = {var_2: var_3}
    var_5 = 'always'
    var_6 = module_0.generate_context(var_1, var_4)
    var_7 = 0
    var_8 = 'Invalid default received'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_context_success_with_overwrites. Retrieved 20/30 statements.
# Partially parsed test_generate_context_invalid_json_raises_exception. Retrieved 3/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_config.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'Original'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Default'
    var_7 = {var_1: var_6}
    var_8 = 'new_var'
    var_9 = '2.0'
    var_10 = 'added'
    var_11 = {var_2: var_9, var_8: var_10}
    var_12 = module_0.generate_context(var_0, var_7, var_11)
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'new_var'
    var_16 = 'Default'
    var_17 = '2.0'
    var_18 = 'added'
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = bool(var_9)
    assert var_20 is True
    var_21 = 'test_config'
    var_22 = bool('test_config' in var_12)
    assert var_22 is True
    var_23 = var_12['test_config']['project_name']
    assert var_23 == 'Default'
    var_24 = var_12['test_config']['version']
    assert var_24 == '2.0'
    var_25 = var_12['test_config']['new_var']
    assert var_25 == 'added'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bad_config.json'
    var_1 = '{ invalid json content ]'
    var_2 = module_0.generate_context(var_0)
    var_3 = 'JSON decoding error'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Ensures that the predicate 'if default_context:' evaluates to False."
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.generate_context(var_1, var_5)
    var_7 = 'cookiecutter'
    var_8 = bool('cookiecutter' in var_6)
    assert var_8 is True
    var_9 = var_6['cookiecutter']['project_name']
    assert var_9 == 'test_project'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_context_skips_default_context_overwrites_when_none. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'test_context'
    var_7 = bool('test_context' in var_5)
    assert var_7 is True
    var_8 = var_5['test_context']['project_name']
    assert var_8 == 'test_project'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_context_with_default_context_triggers_predicate. Retrieved 8/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Ensure that the predicate 'if default_context:' evaluates to True."
    var_1 = 'test_config.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'overridden_name'
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_1, var_6)
    var_8 = 'test_config'
    var_9 = bool('test_config' in var_7)
    assert var_9 is True
    var_10 = var_7['test_config']['project_name']
    assert var_10 == 'overridden_name'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_files_accept_hooks_false. Retrieved 6/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = '/tmp/repo'
    var_3 = {}
    var_4 = False
    var_5 = module_0.generate_files(var_2, var_3, accept_hooks=var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 7/18 statements.
# Partially parsed test_render_and_create_dir_raises_exception_if_exists_and_no_overwrite. Retrieved 4/15 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 4/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

import pathlib as module_0

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)
    var_8 = 'my_project'
    var_9 = bool(var_2)
    assert var_9 is True

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = False

def test_case_0():
    var_0 = 'overwrite_me'
    var_1 = {}
    var_2 = 'overwrite_me'
    var_3 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_already_exists_no_overwrite_raises_exception. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_already_exists_with_overwrite_success. Retrieved 6/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

import pathlib as module_0

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'
    var_5 = 'my_project'
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = False

import pathlib as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = 'project'
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = True



# Parsed testcases at query #3
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
    var_9 = var_4['name']
    assert var_9 == 'new'
    var_10 = var_4['version']
    assert var_10 == 2

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = 'new_var'
    var_8 = bool('new_var' not in var_2)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'theme'
    var_4 = 'dark'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)
    var_9 = var_2['settings']['theme']
    assert var_9 == 'dark'

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
    var_9 = var_5['choices']
    var_10 = bool(var_5['choices'] == ['a', 'c'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'z'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'option'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = 'three'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['option'][0]
    assert var_8 == 'two'
    var_9 = 'one'
    var_10 = bool('one' in var_5['option'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'option'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'three'
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
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_0 = 'database'
    var_1 = 'host'
    var_2 = 'port'
    var_3 = 'localhost'
    var_4 = 5432
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'user'
    var_8 = 9999
    var_9 = 'admin'
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['database']['host']
    assert var_13 == 'localhost'
    var_14 = var_6['database']['port']
    assert var_14 == 9999
    var_15 = var_6['database']['user']
    assert var_15 == 'admin'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['items']
    var_12 = bool(var_4['items'] == ['c', 'd'])
    assert var_12 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'nested_key'
    var_2 = 'original_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'not_a_dict'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = var_4['my_var']
    assert var_8 == 'not_a_dict'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0
import collections as module_1

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "0.1.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'overwritten_name'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '1.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = 'cookiecutter'
    var_11 = (var_10, var_9)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.OrderedDict(*var_13, **var_14)
    var_16 = bool(var_8 == var_15)
    assert var_16 is True
    var_17 = var_8['cookiecutter']['project_name']
    assert var_17 == 'overwritten_name'
    var_18 = var_8['cookiecutter']['version']
    assert var_18 == '1.0.0'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "incomplete"'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'JSON decoding error while loading'
    var_4 = 'Decoding error details'
    var_5 = 'ContextDecodingException not raised'
    var_6 = AssertionError(var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['cookiecutter']['project_name']
    assert var_3 == 'original'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 18/32 statements.


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
    var_11 = "_run_hook_from_repo_dir' function is deprecated"
    var_12 = 'repo'
    var_13 = 'post_gen_project'
    var_14 = 'project'
    var_15 = 'foo'
    var_16 = 'bar'
    var_17 = {var_15: var_16}
    var_18 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_apply_overwrites_to_context_invalid_bool_response. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_error. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_error_if_exists_and_no_overwrite. Retrieved 2/7 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = '/tmp'

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'world'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter_test'
    var_4 = '{{name}}'
    var_5 = '/tmp/test_output'
    var_6 = False
    var_7 = '/tmp/test_output/rendered_name'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test_output'

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'world'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter_test'
    var_4 = '{{name}}'
    var_5 = '/tmp/test_output'
    var_6 = True
    var_7 = '/tmp/test_output/rendered_name'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)



# Parsed testcases at query #9
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
    var_10 = bool(var_4 == {'name': 'new', 'version': 1})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = bool(var_2 == {'name': 'old'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'theme'
    var_4 = 'dark'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)
    var_9 = bool(var_2 == {'settings': {'theme': 'dark'}})
    assert var_9 is True

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
    var_9 = var_5['choices']
    var_10 = bool(var_5['choices'] == ['a', 'c'])
    assert var_10 is True

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
    var_0 = 'option'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = 'three'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['option']
    var_9 = bool(var_5['option'] == ['two', 'one', 'three'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'option'
    var_1 = 'one'
    var_2 = 'two'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'three'
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
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is False

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
    var_0 = 'meta'
    var_1 = 'version'
    var_2 = 'tags'
    var_3 = 1
    var_4 = 'old'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'author'
    var_9 = 'new'
    var_10 = [var_9]
    var_11 = 'dev'
    var_12 = {var_2: var_10, var_8: var_11}
    var_13 = {var_0: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_7, var_13)
    var_15 = var_7['meta']
    var_16 = bool(var_7['meta'] == {'version': 1, 'tags': ['new'], 'author': 'dev'})
    assert var_16 is True

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
    var_10 = var_4['items']
    var_11 = bool(var_4['items'] == ['c'])
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_skips_creation_when_dir_exists. Retrieved 11/23 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter_test'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = 'test_dir'
    var_6 = 'rendered_dir'
    var_7 = var_4 / var_6
    var_8 = True
    var_9 = 'non_existent_path'
    var_10 = var_4 / var_9
    var_11 = module_1.rmtree(var_10)
    var_12 = False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 4/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = "{'broken': 'json',"
    var_2 = 'utf-8'
    var_3 = module_0.generate_context(var_0)
    var_4 = 'JSON decoding error while loading'



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
    var_7 = 'admin'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    assert var_11 == 1
    var_12 = 'author'
    var_13 = bool('author' not in var_4)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'notifications'
    var_3 = 'dark'
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_key'
    var_8 = 'light'
    var_9 = 'val'
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_4)
    var_13 = var_6['settings']['theme']
    assert var_13 == 'light'
    var_14 = var_6['settings']['notifications']
    assert var_14 is True
    var_15 = var_6['settings']['new_key']
    assert var_15 == 'val'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'languages'
    var_1 = 'python'
    var_2 = 'rust'
    var_3 = 'go'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['languages']
    var_10 = bool(var_5['languages'] == ['python', 'go'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'languages'
    var_1 = 'python'
    var_2 = 'rust'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'cpp'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = var_4['mode'][0]
    assert var_7 == 'slow'
    var_8 = var_4['mode'][1]
    assert var_8 == 'fast'

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
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['debug']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['debug']
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
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
    var_13 = module_0.apply_overwrites_to_context(var_6, var_12)
    var_14 = var_6['a']['b']['c']
    assert var_14 == 2
    var_15 = var_6['a']['b']['d']
    assert var_15 == 3

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'x'
    var_6 = 'y'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['items']
    var_12 = bool(var_4['items'] == ['x', 'y'])
    assert var_12 is True



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
    var_5 = 'test.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_failure. Retrieved 6/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['enabled']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_context.json'
    var_4 = module_0.generate_context(var_3)
    var_5 = 'test_context'
    var_6 = bool('test_context' in var_4)
    assert var_6 is True
    var_7 = var_4['test_context']['project_name']
    assert var_7 == 'test_project'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_successful_creation. Retrieved 7/19 statements.
# Partially parsed test_render_and_create_dir_raises_if_exists_without_overwrite. Retrieved 6/20 statements.
# Partially parsed test_render_and_create_dir_returns_false_if_exists_with_overwrite. Retrieved 4/17 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/test'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'project_{{ name }}'
    var_5 = 'name'
    var_6 = 'foo'
    var_7 = {var_5: var_6}
    var_8 = False

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = False
    var_4 = 'Expected OutputDirExistsException'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_context_reads_file_successfully. Retrieved 11/14 statements.


import cookiecutter.generate as module_0
import collections as module_1

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)
    var_7 = 'test_context'
    var_8 = (var_7, var_5)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.OrderedDict(*var_10, **var_11)
    var_13 = bool(var_6 == var_12)
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_successful_file_open. Retrieved 5/8 statements.
# Partially parsed test_generate_context_valid_json_structure. Retrieved 5/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'test_context'
    var_6 = bool('test_context' in var_4)
    assert var_6 is True
    var_7 = var_4['test_context']
    var_8 = bool(var_4['test_context'] == var_3)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = '{"project_name": "my_project", "version": "0.1.0"}'
    var_2 = module_0.generate_context(var_0)
    var_3 = 'valid'
    var_4 = var_2[var_3]
    var_5 = var_2['valid']['project_name']
    assert var_5 == 'my_project'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 7/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 38 evaluates to False when default_context is None.'
    var_1 = 'test_cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.generate_context(var_1, var_5)
    var_7 = 'test_cookiecutter'
    var_8 = bool('test_cookiecutter' in var_6)
    assert var_8 is True
    var_9 = var_6['test_cookiecutter']
    var_10 = bool(var_6['test_cookiecutter'] == var_4)
    assert var_10 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_true_when_dir_exists. Retrieved 13/23 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = {}
    var_7 = 'template_name'
    var_8 = '.'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)
    var_12 = False
    var_13 = 'rendered_name'
    var_14 = [var_11, var_13]
    var_15 = {}
    var_16 = module_0.Path(*var_14, **var_15)
    var_17 = var_16.mkdir(parents=var_4, exist_ok=var_4)
    var_18 = module_1.rmtree(var_16)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_context_with_default_context_triggers_if_statement. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overwritten_project'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_context'
    var_8 = bool('test_context' in var_6)
    assert var_8 is True
    var_9 = var_6['test_context']['project_name']
    assert var_9 == 'overwritten_project'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/12 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 12/25 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 6/13 statements.
# Partially parsed test_generate_file_empty_name_is_directory. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/binary.dat'
    var_2 = {}
    var_3 = 'binary.dat'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/script.py'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'script.py'
    var_10 = 'template/script.py'
    var_11 = "print('value')"

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/exists.txt'
    var_2 = {}
    var_3 = True
    var_4 = 'The resulting file already exists: %s'
    var_5 = 'exists.txt'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/folder_tmpl'
    var_2 = {}
    var_3 = 'The resulting file name is empty: %s'
    var_4 = 'folder_tmpl'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true. Retrieved 9/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that the predicate 'if default_context:' evaluates to True."
    var_1 = 'test_context.json'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'overridden_name'
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.generate_context(var_1, var_6, var_7)
    var_9 = 'test_context'
    var_10 = bool('test_context' in var_8)
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_none. Retrieved 3/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter-test'
    var_2 = ''

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter-test'
    var_2 = None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_render_and_create_dir_path_already_exists. Retrieved 13/24 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_output_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'rendered_dir'
    var_7 = var_3 / var_6
    var_8 = {}
    var_9 = 'template_{{ name }}'
    var_10 = 'name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = var_3.rmdir()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True

def test_case_0():
    var_0 = '{"project_name": "missing_quote}'
    var_1 = 'cookiecutter.json'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_and_create_dir_path_already_exists. Retrieved 13/25 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_output_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'rendered_name'
    var_7 = var_3 / var_6
    var_8 = {}
    var_9 = 'template_{{ name }}'
    var_10 = 'name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = module_1.rmtree(var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 18/23 statements.
# Partially parsed test_generate_files_copy_only_logic. Retrieved 13/18 statements.


import pathlib as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/output'
    var_2 = '/fake/repo/{{cookiecutter.project_name}}'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_copy_without_render'
    var_9 = 'my_project'
    var_10 = []
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = '.'
    var_14 = []
    var_15 = 'file1.txt'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = module_1.generate_files(var_0, var_12, var_1)
    var_19 = '/fake/output/my_project'
    var_20 = bool(var_18 == var_11)
    assert var_20 is True

import pathlib as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/template'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = None
    var_6 = module_1.generate_files(var_0, var_5)
    assert var_6 == '/fake/project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'err'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_files(var_0, var_5)
    var_7 = 'Unable to create project directory'
    var_8 = 'Did not raise UndefinedVariableInTemplate'
    var_9 = AssertionError(var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.bin'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = '.'
    var_8 = []
    var_9 = 'data.bin'
    var_10 = [var_9]
    var_11 = (var_7, var_8, var_10)
    var_12 = module_0.generate_files(var_0, var_6)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_context_default_context_is_none. Retrieved 6/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'test_context'
    var_7 = bool('test_context' in var_5)
    assert var_7 is True
    var_8 = var_5['test_context']['project_name']
    assert var_8 == 'test_project'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_render_and_create_dir_enters_overwrite_logic. Retrieved 10/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'project_name'
    var_5 = var_3 / var_4
    var_6 = {}
    var_7 = True
    var_8 = 'project_{% upper %}name{% end %}'
    var_9 = 'upper'
    var_10 = 'lambda x: x.upper()'
    var_11 = {var_9: var_10}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_render_and_create_dir_skips_creation_when_dir_exists_and_overwrite_is_true. Retrieved 9/19 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_output_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'rendered_name'
    var_7 = var_3 / var_6
    var_8 = 'test_dir'
    var_9 = {}
    var_10 = module_1.rmtree(var_3)



