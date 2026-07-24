####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 18/32 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_correct_function. Retrieved 7/11 statements.


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
    var_11 = "The '_run_hook_from_repo_dir' function is deprecated"
    var_12 = 'repo'
    var_13 = 'post_gen_project'
    var_14 = 'project'
    var_15 = 'foo'
    var_16 = 'bar'
    var_17 = {var_15: var_16}
    var_18 = True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = {}



# Parsed testcases at query #2
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
    var_0 = '{"project_name": "old_name", "use_git": false}'
    var_1 = 'project_name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = 'use_git'
    var_5 = 'new_name'
    var_6 = 'yes'
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = 'cookiecutter.json'
    var_9 = module_0.generate_context(var_8, var_3, var_7)
    var_10 = var_9['cookiecutter']['project_name']
    assert var_10 == 'new_name'
    var_11 = var_9['cookiecutter']['use_git']
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "missing_quote}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'JSON decoding error'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 12/18 statements.
# Partially parsed test_render_and_create_dir_fails_if_exists_without_overwrite. Retrieved 9/17 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 7/14 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = './test_output_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'template_{{ name }}'
    var_7 = 'name'
    var_8 = 'user'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 'rendered_name'
    var_12 = [var_3, var_11]
    var_13 = {}
    var_14 = module_0.Path(*var_12, **var_13)
    var_15 = module_1.rmtree(var_3)

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = './test_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'existing_dir'
    var_5 = var_3 / var_4
    var_6 = True
    var_7 = 'existing_dir'
    var_8 = {}
    var_9 = False
    var_10 = 'already exists'
    var_11 = module_1.rmtree(var_3)

import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = './test_overwrite'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'overwrite_me'
    var_5 = var_3 / var_4
    var_6 = True
    var_7 = {}
    var_8 = module_1.rmtree(var_3)



# Parsed testcases at query #4
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
    var_0 = 'config'
    var_1 = 'debug'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'logging'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_6)
    var_10 = var_4['config']['debug']
    assert var_10 is False
    var_11 = var_4['config']['logging']
    assert var_11 is True

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
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = 'medium'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['mode'][0]
    assert var_8 == 'slow'
    var_9 = 'fast'
    var_10 = bool('fast' in var_5['mode'])
    assert var_10 is True

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
    var_0 = 'settings'
    var_1 = 'modes'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'c'
    var_8 = [var_7]
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.apply_overwrites_to_context(var_6, var_10, in_dictionary_variable=var_11)
    var_13 = var_6['settings']['modes']
    var_14 = bool(var_6['settings']['modes'] == ['c'])
    assert var_14 is True

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
    var_15 = var_6['a']['b']['c']
    assert var_15 == 2
    var_16 = var_6['a']['b']['d']
    assert var_16 == 3



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice_var'][0]
    assert var_8 == 'option2'



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
    var_9 = 'cookiecutter'
    var_10 = var_8[var_9]['project_name']
    assert var_10 == 'default_project'
    var_11 = var_8[var_9]['version']
    assert var_11 == '2.0.0'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'config.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['config']['key']
    assert var_3 == 'value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'bad.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 'empty.json'
    var_2 = None
    var_3 = module_0.generate_context(var_1, var_2, var_2)
    var_4 = var_3['empty']
    var_5 = bool(var_3['empty'] == {})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"a": "orig", "b": "orig"}'
    var_1 = 'test.json'
    var_2 = 'a'
    var_3 = 'default'
    var_4 = {var_2: var_3}
    var_5 = 'c'
    var_6 = 'extra'
    var_7 = 'new'
    var_8 = {var_2: var_6, var_5: var_7}
    var_9 = module_0.generate_context(var_1, var_4, var_8)
    var_10 = 'test'
    var_11 = var_9[var_10]
    var_12 = var_11['a']
    assert var_12 == 'extra'
    var_13 = var_11['b']
    assert var_13 == 'orig'



# Parsed testcases at query #7
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
    var_6 = 'script.py'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False

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
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'FILE.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.is_copy_only_path(var_2, var_5)
    assert var_6 is True



# Parsed testcases at query #8
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
    var_0 = '{"invalid_json": '
    var_1 = 'cookiecutter.json'



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
    var_0 = 'mode'
    var_1 = 'fast'
    var_2 = 'slow'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = var_4['mode']
    var_8 = bool(var_4['mode'] == ['slow', 'fast'])
    assert var_8 is True

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
    var_0 = 'settings'
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = False
    var_4 = 8080
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'user'
    var_8 = 'true'
    var_9 = 'admin'
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['settings']['debug']
    assert var_13 is True
    var_14 = var_6['settings']['port']
    assert var_14 == 8080
    var_15 = var_6['settings']['user']
    assert var_15 == 'admin'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'modes'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_3]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_6, var_9, in_dictionary_variable=var_10)
    var_12 = var_6['settings']['modes']
    var_13 = bool(var_6['settings']['modes'] == ['b'])
    assert var_13 is True

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
    var_3 = 'n'
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
    var_0 = 'root'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)
    var_9 = var_2['root']['new_key']
    assert var_9 == 'value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = 'new_var'
    var_9 = bool('new_var' not in var_2)
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 8/18 statements.
# Partially parsed test_render_and_create_dir_raises_error_if_exists_and_no_overwrite. Retrieved 4/13 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 6/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test'
    var_2 = ''

import pathlib as module_0

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/test'
    var_5 = '/tmp/test/my_project'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/test'
    var_3 = False

import pathlib as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = '/tmp/test/existing_dir'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_context_preserves_order. Retrieved 5/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_name'
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
    var_3 = 'JSON decoding error'
    var_4 = 'ContextDecodingException not raised'
    var_5 = AssertionError(var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = 'test.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'test'
    var_4 = var_2[var_3]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"existing": "value"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'new_var'
    var_3 = 'should_not_appear'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, var_4)
    var_6 = 'existing'
    var_7 = bool('existing' in var_5['cookiecutter'])
    assert var_7 is True
    var_8 = 'new_var'
    var_9 = bool('new_var' not in var_5['cookiecutter'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "original"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'key'
    var_3 = 'default'
    var_4 = {var_2: var_3}
    var_5 = 'extra'
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_1, var_4, var_6)
    var_8 = var_7['cookiecutter']['key']
    assert var_8 == 'extra'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overwritten_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_context'
    var_8 = bool('test_context' in var_6)
    assert var_8 is True
    var_9 = var_6['test_context']['project_name']
    assert var_9 == 'overwritten_name'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 6/16 statements.
# Partially parsed test_render_and_create_dir_raises_error_if_exists_and_no_overwrite. Retrieved 5/17 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 5/16 statements.


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
    var_8 = bool(var_1)
    assert var_8 is True

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_files_basic_rendering. Retrieved 9/24 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 17/38 statements.
# Partially parsed test_generate_files_directory_rendering. Retrieved 12/34 statements.
# Partially parsed test_generate_files_overwrite_logic. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'cookiecutter-test-project'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"}'
    var_3 = 'utf-8'
    var_4 = 'Hello {{ project_name }}!'
    var_5 = 'hello.txt'
    var_6 = 'project_name'
    var_7 = 'world'
    var_8 = {var_6: var_7}

import json as module_0

def test_case_0():
    var_0 = 'cookiecutter-test-project'
    var_1 = 'project_name'
    var_2 = 'cookiecutter'
    var_3 = 'my_project'
    var_4 = '_copy_without_render'
    var_5 = 'static/*'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_1: var_3, var_2: var_7}
    var_9 = 'cookiecutter.json'
    var_10 = {}
    var_11 = module_0.dumps(var_8, **var_10)
    var_12 = 'utf-8'
    var_13 = 'static'
    var_14 = 'data.txt'
    var_15 = '{{ project_name }}'
    var_16 = 'world'
    var_17 = {var_1: var_16}

def test_case_0():
    var_0 = 'cookiecutter-test-project'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"}'
    var_3 = 'utf-8'
    var_4 = '{{ project_name }}_dir'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'utf-template'
    var_8 = 'project_name'
    var_9 = 'my_app'
    var_10 = {var_8: var_9}
    var_11 = 'my_app_dir'

def test_case_0():
    var_0 = 'cookiecutter-test-project'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"}'
    var_3 = 'utf-8'
    var_4 = 'file.txt'
    var_5 = 'new content'
    var_6 = 'project_name'
    var_7 = 'world'
    var_8 = {var_6: var_7}
    var_9 = 'my_project'
    var_10 = 'old content'
    var_11 = False
    var_12 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/11 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 11/23 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 4/10 statements.
# Partially parsed test_generate_file_empty_filename_is_directory. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/binary.bin'
    var_2 = {}
    var_3 = '/tmp/output/binary.bin'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/script.py'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'test'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'template/script.py'
    var_10 = "print('hello name')"

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/existing.txt'
    var_2 = {}
    var_3 = True

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/dir_as_file.txt'
    var_2 = {}



# Parsed testcases at query #16
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
    var_0 = '{"project_name": "my_project", invalid_json}'
    var_1 = 'cookierunner.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "original"}'
    var_1 = 'config.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['config']['project_name']
    assert var_3 == 'original'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"existing": "value"}'
    var_1 = 'test.json'
    var_2 = 'new_key'
    var_3 = 'should_not_appear'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, var_4)
    var_6 = 'new_key'
    var_7 = bool('new_key' not in var_5['test'])
    assert var_7 is True
    var_8 = var_5['test']['existing']
    assert var_8 == 'value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "old_value"}'
    var_1 = 'test.json'
    var_2 = 'key'
    var_3 = 'new_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, extra_context=var_4)
    var_6 = var_5['test']['key']
    assert var_6 == 'new_value'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_files_basic_creation. Retrieved 13/29 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 14/28 statements.


def test_case_0():
    var_0 = '{{cookiecutter.project_slug}}'
    var_1 = 'Hello {{cookiecutter.name}}!'
    var_2 = 'hello.txt'
    var_3 = 'cookiecutter'
    var_4 = 'project_slug'
    var_5 = 'name'
    var_6 = 'my_project'
    var_7 = 'World'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = '{{cookiecutter.project_slug}}'
    var_11 = '{"cookiecutter": {"project_slug": "my_project", "name": "World"}}'
    var_12 = 'hello.txt'

def test_case_0():
    var_0 = '{{cookiecutter.project_slug}}'
    var_1 = b'\x00\x01\x02\x03'
    var_2 = 'data.bin'
    var_3 = '{{cookiecutter.project_slug}}'
    var_4 = '{"cookiecutter": {"project_slug": "my_project", "_copy_without_render": ["*.bin"]}}'
    var_5 = 'cookiecutter'
    var_6 = 'project_slug'
    var_7 = '_copy_without_render'
    var_8 = 'my_project'
    var_9 = '*.bin'
    var_10 = [var_9]
    var_11 = {var_6: var_8, var_7: var_10}
    var_12 = {var_5: var_11}
    var_13 = 'data.bin'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_file_predicate_true. Retrieved 17/25 statements.


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
    var_11 = 'os.path.isdir'
    var_12 = False
    var_13 = 'os.path.exists'
    var_14 = 'is_binary'
    var_15 = 'builtins.open'
    var_16 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_and_create_dir_output_dir_exists_true. Retrieved 11/22 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_output_root'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'existing_dirname'
    var_7 = var_3 / var_6
    var_8 = {}
    var_9 = 'existing_dirname'
    var_10 = str(var_3)
    var_11 = False
    var_12 = module_1.rmtree(var_3)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overridden_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_cookiecutter'
    var_8 = bool('test_cookiecutter' in var_6)
    assert var_8 is True
    var_9 = var_6['test_cookiecutter']['project_name']
    assert var_9 == 'overridden_name'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_on_empty_dirname. Retrieved 4/10 statements.


import pathlib as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 13/21 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 5/13 statements.
# Partially parsed test_generate_context_with_nested_dict_overwrites. Retrieved 13/19 statements.


import json as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '0.1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'default_project'
    var_9 = {var_1: var_8}
    var_10 = 'new_var'
    var_11 = '1.0.0'
    var_12 = 'new_value'
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = 'cookiecutter'
    var_15 = 'new_var'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bad.json'
    var_1 = '{ invalid json'
    var_2 = module_0.generate_context(var_0)
    var_3 = 'JSON decoding error'
    var_4 = 'ContextDecodingException not raised'
    var_5 = AssertionError(var_4)

import json as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'settings'
    var_2 = 'debug'
    var_3 = 'port'
    var_4 = False
    var_5 = 8080
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = 'true'
    var_11 = 9000
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = {var_1: var_12}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_file_is_binary_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test_binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_fails. Retrieved 6/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/10 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 10/19 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 4/9 statements.
# Partially parsed test_generate_file_empty_output_path_is_dir. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/binary.bin'
    var_2 = {}
    var_3 = '/tmp/output/binary.bin'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/text.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'world'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'template/text.txt'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/exists.txt'
    var_2 = {}
    var_3 = True

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/dir_template'
    var_2 = {}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_file_template_syntax_error_raises_exception. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'Syntax Error'
    var_6 = 1



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 9/22 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_target. Retrieved 9/12 statements.


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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'post_gen_project'
    var_2 = 'project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}



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
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = False
    var_4 = 80
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'timeout'
    var_8 = 'true'
    var_9 = 30
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['config']['debug']
    assert var_13 is True
    var_14 = var_6['config']['port']
    assert var_14 == 80
    var_15 = var_6['config']['timeout']
    assert var_15 == 30

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'small'
    var_2 = 'medium'
    var_3 = 'large'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['type']
    var_9 = bool(var_5['type'] == ['medium', 'small', 'large'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'small'
    var_2 = 'medium'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'large'
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
    var_9 = var_5['features']
    var_10 = bool(var_5['features'] == ['auth', 'cache'])
    assert var_10 is True

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'logging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'cache'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_success_overwrite. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_error_empty_name. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_error_directory_exists_no_overwrite. Retrieved 4/9 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'my_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/cookiecutter'
    var_5 = False
    var_6 = '/tmp/cookiecutter/my_project'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = [var_6]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)

import pathlib as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = True
    var_4 = '/tmp/cookiecutter/existing_dir'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'

def test_case_0():
    var_0 = 'fixed_name'
    var_1 = {}
    var_2 = '/tmp/cookiecutter'
    var_3 = False



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['is_enabled']
    assert var_6 is True



# Parsed testcases at query #5
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
    var_9 = bool(var_4 == {'name': 'new', 'version': 2})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'name': 'original'})
    assert var_7 is True

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
    var_11 = bool(var_4 == {'settings': {'theme': 'dark', 'font': 'arial'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = 'db'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['features']
    var_10 = bool(var_5['features'] == ['auth', 'db'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid'
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
    var_7 = var_4['mode']
    var_8 = bool(var_4['mode'] == ['slow', 'fast'])
    assert var_8 is True

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
    var_3 = 'not-a-bool'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'db'
    var_2 = 'debug'
    var_3 = 'host'
    var_4 = 'port'
    var_5 = 'localhost'
    var_6 = 5432
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = True
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = '127.0.0.1'
    var_12 = {var_3: var_11}
    var_13 = 'false'
    var_14 = {var_1: var_12, var_2: var_13}
    var_15 = {var_0: var_14}
    var_16 = module_0.apply_overwrites_to_context(var_10, var_15)
    var_17 = var_10['config']['db']['host']
    assert var_17 == '127.0.0.1'
    var_18 = var_10['config']['db']['port']
    assert var_18 == 5432
    var_19 = var_10['config']['debug']
    assert var_19 is False

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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_calls_correct_function. Retrieved 9/14 statements.
# Partially parsed test_run_hook_from_repo_dir_emits_deprecation_warning. Retrieved 7/18 statements.


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
    var_0 = 'always'
    var_1 = 'repo'
    var_2 = 'post_gen_project'
    var_3 = 'project'
    var_4 = {}
    var_5 = False
    var_6 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)
    var_7 = "The '_run_hook_from_repo_dir' function is deprecated"



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 9/25 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 8/22 statements.
# Partially parsed test_generate_files_empty_context_error. Retrieved 5/18 statements.


def test_case_0():
    var_0 = '{{project_slug}}'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_slug": "my_project"}'
    var_3 = 'hello.txt'
    var_4 = 'Hello {{project_slug}}'
    var_5 = 'project_slug'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = 'cookiecutter_{{project_slug}}'

def test_case_0():
    var_0 = 'cookiecutter_test_{{project_slug}}'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_slug": "my_project", "_copy_without_render": ["*.txt"]}'
    var_3 = 'keep_me.txt'
    var_4 = 'Value: {{project_slug}}'
    var_5 = 'project_slug'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = '{{project_slug}}'

def test_case_0():
    var_0 = 'cookiecutter_test'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_slug": "my_project"}'
    var_3 = 'error.txt'
    var_4 = '{{non_existent_variable}}'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 8/13 statements.
# Partially parsed test_generate_context_with_overwrites. Retrieved 12/16 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 5/11 statements.


import json as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'old_name'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'default_name'
    var_9 = {var_1: var_8}
    var_10 = 'extra_name'
    var_11 = '2.0.0'
    var_12 = {var_1: var_10, var_2: var_11}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bad.json'
    var_1 = '{ invalid json }'
    var_2 = module_0.generate_context(var_0)
    var_3 = 'Should have raised ContextDecodingException'
    var_4 = AssertionError(var_3)
    var_5 = 'JSON decoding error'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 6/11 statements.
# Partially parsed test_render_and_create_dir_success_overwrite_existing. Retrieved 8/15 statements.
# Partially parsed test_render_and_create_dir_raises_error_on_existing_without_overwrite. Retrieved 7/15 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/out'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}_dir'
    var_4 = 'outputs'
    var_5 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_project'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'
    var_5 = 'outputs'
    var_6 = module_0.Environment()
    var_7 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'collision'
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = '{{ name }}'
    var_4 = 'outputs'
    var_5 = module_0.Environment()
    var_6 = False



# Parsed testcases at query #5
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
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'font'
    var_6 = 'sans'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['config']['theme']
    assert var_11 == 'dark'
    var_12 = var_4['config']['font']
    assert var_12 == 'sans'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = 'db'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['features']
    var_10 = bool(var_5['features'] == ['auth', 'db'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'api'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'ldap'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'small'
    var_2 = 'medium'
    var_3 = 'large'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['type'][0]
    assert var_8 == 'medium'
    var_9 = var_5[var_0]
    var_10 = len(var_9)
    assert var_10 == 3

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'small'
    var_2 = 'medium'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'large'
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
    var_0 = 'settings'
    var_1 = 'logging'
    var_2 = 'level'
    var_3 = 'format'
    var_4 = 'INFO'
    var_5 = 'text'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'DEBUG'
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_8, var_12)
    var_14 = var_8['settings']['logging']['level']
    assert var_14 == 'DEBUG'
    var_15 = var_8['settings']['logging']['format']
    assert var_15 == 'text'

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
    var_10 = var_4['tags']
    var_11 = bool(var_4['tags'] == ['c', 'd'])
    assert var_11 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_4 = 'ignored'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = 'new_var'
    var_8 = bool('new_var' not in var_2)
    assert var_8 is True

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
    var_0 = 'mode'
    var_1 = 'debug'
    var_2 = 'release'
    var_3 = 'test'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_3}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['mode']
    var_9 = bool(var_5['mode'] == ['test', 'debug', 'release'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'mode'
    var_1 = 'debug'
    var_2 = 'release'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'production'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

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
    var_9 = 'admin'
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['db']['host']
    assert var_13 == 'localhost'
    var_14 = var_6['db']['port']
    assert var_14 == 9999
    var_15 = var_6['db']['user']
    assert var_15 == 'admin'

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
    var_0 = 'settings'
    var_1 = 'modes'
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
    var_14 = var_6['settings']['modes']
    var_15 = bool(var_6['settings']['modes'] == ['c', 'd'])
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 8/14 statements.
# Partially parsed test_render_and_create_dir_success_overwrite_existing. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_raises_error_on_existing_without_overwrite. Retrieved 7/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/output'
    var_2 = ''

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = '/tmp/output/my_project'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = True
    var_9 = '/tmp/output/my_project'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{cookiecutter.name}}'
    var_8 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 12/16 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "test_project", "version": "1.0.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_name'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = var_8[var_12]

def test_case_0():
    var_0 = '{"project_name": "test_project", }'
    var_1 = 'cookietytter.json'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_generate_context_predicate_true_with_default_context. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_config.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overridden_project'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_config'
    var_8 = bool('test_config' in var_6)
    assert var_8 is True
    var_9 = var_6['test_config']['project_name']
    assert var_9 == 'overridden_project'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_none. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_triggers_overwrite_logic_when_path_exists. Retrieved 13/24 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = './test_temp_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'rendered_name'
    var_7 = var_3 / var_6
    var_8 = False
    var_9 = {}
    var_10 = 'template_name'
    var_11 = var_3.absolute()
    var_12 = str(var_11)
    var_13 = True
    var_14 = module_1.rmtree(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/5 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 9/14 statements.
# Failed to parse test_render_and_create_dir_raises_error_if_exists_and_no_overwrite.
# Partially parsed test_render_and_create_dir_returns_correct_bool_when_overwriting. Retrieved 1/6 statements.


def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = '/tmp/cookiecutter'

import pathlib as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/cookiecutter'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = '{{ cookiecutter_name }}'
    var_8 = False
    var_9 = '/tmp/cookiecutter/project_name'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)

def test_case_0():
    var_0 = {}



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean-value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_enters_overwrite_logic_when_dir_exists. Retrieved 7/16 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'template_name'
    var_5 = {}
    var_6 = 'rendered_name'
    var_7 = var_3 / var_6
    var_8 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 10/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "test_project", "version": "1.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = {var_2: var_3, var_5: var_6}
    var_10 = var_8['cookiecutter']
    var_11 = bool(var_8['cookiecutter'] == var_9)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"invalid": json'
    var_1 = 'cookierunner.json'
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'test_cookiecutter'
    var_6 = bool('test_cookiecutter' in var_4)
    assert var_6 is True
    var_7 = var_4['test_cookiecutter']['project_name']
    assert var_7 == 'test_project'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_empty_name_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_success_new_directory. Retrieved 5/17 statements.
# Partially parsed test_render_and_create_dir_error_if_exists_no_overwrite. Retrieved 6/17 statements.
# Partially parsed test_render_and_create_dir_success_with_overwrite. Retrieved 6/16 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp/test'

def test_case_0():
    var_0 = 'template_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'

def test_case_0():
    var_0 = 'template_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'
    var_5 = False

def test_case_0():
    var_0 = 'template_{{ name }}'
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'
    var_5 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 12/26 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_correct_function. Retrieved 9/13 statements.


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
    var_11 = "The '_run_hook_from_repo_dir' function is deprecated"
    var_12 = {var_4: var_5}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'output_dir'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_successfully_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'test_context'
    var_6 = bool('test_context' in var_4)
    assert var_6 is True
    var_7 = var_4['test_context']['project_name']
    assert var_7 == 'test_project'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_apply_overwrites_to_context_bool_conversion_invalid_response. Retrieved 6/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'not-a-boolean'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'could not be converted to a boolean'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 6/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'test_cookiecutter'
    var_7 = bool('test_cookiecutter' in var_5)
    assert var_7 is True
    var_8 = var_5['test_cookiecutter']['project_name']
    assert var_8 == 'test_project'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 4/12 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 11/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 4/10 statements.
# Partially parsed test_generate_file_empty_outfile_name. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template/binary.dat'
    var_2 = {}
    var_3 = 'binary.dat'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template/config.j2'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'my_project'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'template/config.j2'
    var_10 = "print('hello')"

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template/existing.txt'
    var_2 = {}
    var_3 = True

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template/{invalid_path}.txt'
    var_2 = 'invalid_path'
    var_3 = ''
    var_4 = {var_2: var_3}



# Parsed testcases at query #18
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
    var_10 = bool(var_4 == {'name': 'new', 'version': 1, 'author': 'tester'})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'extra'
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
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'z'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = var_4['choice']
    var_8 = bool(var_4['choice'] == ['option2', 'option1'])
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'option2'
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

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
    var_9 = 'nested'
    var_10 = [var_1, var_2]
    var_11 = {var_0: var_10}
    var_12 = {var_9: var_11}
    var_13 = [var_5, var_6]
    var_14 = {var_0: var_13}
    var_15 = {var_9: var_14}
    var_16 = False
    var_17 = module_0.apply_overwrites_to_context(var_12, var_15, in_dictionary_variable=var_16)
    var_18 = var_12['nested']['items']
    var_19 = bool(var_12['nested']['items'] == ['c', 'd'])
    assert var_19 is True

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
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 'role'
    var_3 = 'old'
    var_4 = 'guest'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'active'
    var_8 = 'admin'
    var_9 = True
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = var_6['user']
    var_14 = bool(var_6['user'] == {'name': 'old', 'role': 'admin', 'active': True})
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['config']
    var_11 = bool(var_4['config'] == {'a': 1, 'b': 2})
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test'
    var_2 = ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_config.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overridden_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_config'
    var_8 = bool('test_config' in var_6)
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_enters_overwrite_logic. Retrieved 8/18 statements.


import pathlib as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/cookiecutter_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_dir'
    var_5 = {}
    var_6 = 'rendered_name'
    var_7 = var_3 / var_6
    var_8 = True
    var_9 = module_1.rmtree(var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_context_with_default_context_evaluates_true. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'overridden_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 'test_cookiecutter'
    var_8 = bool('test_cookiecutter' in var_6)
    assert var_8 is True
    var_9 = var_6['test_cookiecutter']['project_name']
    assert var_9 == 'overridden_name'



# Parsed testcases at query #23
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 1.0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = var_4['name']
    assert var_10 == 'new'
    var_11 = var_4['version']
    var_12 = bool(var_4['version'] == 1.0)
    assert var_12 is True
    var_13 = 'author'
    var_14 = bool('author' not in var_4)
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = False
    var_4 = 8080
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_key'
    var_8 = True
    var_9 = 'value'
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_8)
    var_13 = var_6['config']['debug']
    assert var_13 is True
    var_14 = var_6['config']['port']
    assert var_14 == 8080
    var_15 = var_6['config']['new_key']
    assert var_15 == 'value'

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
    var_8 = var_5['color']
    var_9 = bool(var_5['color'] == ['blue', 'red', 'green'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = 'red'
    var_2 = 'blue'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'yellow'
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
    var_10 = var_4['features']
    var_11 = bool(var_4['features'] == ['cache'])
    assert var_11 is True

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
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_6, var_12, in_dictionary_variable=var_13)
    var_15 = var_6['a']['b']['c']
    assert var_15 == 2
    var_16 = var_6['a']['b']['d']
    assert var_16 == 3



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 14/42 statements.


def test_case_0():
    var_0 = 'test_cookiecutter_root'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = 'cookiecutter-{{ project_name }}'
    var_4 = True
    var_5 = 'hello.txt'
    var_6 = 'Hello {{ project_name }}!'
    var_7 = 'project_name'
    var_8 = 'cookiecutter'
    var_9 = 'my_project'
    var_10 = '_new_lines'
    var_11 = '\n'
    var_12 = {var_10: var_11}
    var_13 = {var_7: var_9, var_8: var_12}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/tmp/cookiecutter'
    var_2 = ''



# Parsed testcases at query #26
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
    var_6 = 'image.png'
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



# Parsed testcases at query #27
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/templates/static/*'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/templates/static/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 6/13 statements.
# Partially parsed test_generate_file_text_render_success. Retrieved 11/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 6/12 statements.
# Partially parsed test_generate_file_empty_output_name. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/output/binary.bin'

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/script.py'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\n'
    var_6 = {var_4: var_5}
    var_7 = 'my_proj'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'template/script.py'
    var_10 = "print('hello')"

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/exists.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = True

def test_case_0():
    var_0 = '/tmp/output'
    var_1 = 'template/dir_template/'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "0.1.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '1.0.0'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_1, var_4, var_7)
    var_9 = 'cookiecutter'
    var_10 = var_8[var_9]['project_name']
    assert var_10 == 'default_project'
    var_11 = bool(var_8[var_9]['version'] << '1.0.0' or var_8[var_9]['version'] == '1.0.0')
    assert var_11 is True
    var_12 = var_8[var_9]['version']
    assert var_12 == '1.0.0'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'my_config.test.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'my_config'
    var_4 = bool('my_config' in var_2)
    assert var_4 is True
    var_5 = var_2['my_config']['key']
    assert var_5 == 'value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'bad.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = 'cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = var_2['cookiecutter']['a']
    assert var_3 == 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_files_accept_hooks_is_true. Retrieved 6/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = '/tmp/template'
    var_3 = {}
    var_4 = '/tmp/output'
    var_5 = module_0.generate_files(var_2, var_3, var_4, accept_hooks=var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 6/10 statements.


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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_context_success. Retrieved 12/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"project_name": "my_project", "version": "0.1.0"}'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = 'new_var'
    var_7 = '1.0.0'
    var_8 = 'new_val'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.generate_context(var_1, var_4, var_9)
    var_11 = {var_2: var_3, var_5: var_7, var_6: var_8}
    var_12 = var_10['cookiecutter']
    var_13 = bool(var_10['cookiecutter'] == var_11)
    assert var_13 is True

def test_case_0():
    var_0 = '{"project_name": "my_project", '
    var_1 = 'cookiecutter.json'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_files_os_walk_predicate_is_true. Retrieved 14/20 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = '.'
    var_3 = 'subdir'
    var_4 = [var_3]
    var_5 = 'file1.txt'
    var_6 = [var_5]
    var_7 = (var_2, var_4, var_6)
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = '/tmp/repo'
    var_12 = '/tmp/output'
    var_13 = module_0.generate_files(var_11, var_10, var_12)
    assert var_13 == '/tmp/project'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_context_skips_default_context_application_when_none. Retrieved 6/14 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'test_cookiecutter'
    var_7 = bool('test_cookiecutter' in var_5)
    assert var_7 is True
    var_8 = var_5['test_cookiecutter']['project_name']
    assert var_8 == 'test_project'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_files_line_36_evaluates_to_true. Retrieved 12/18 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/repo/{{cookiecutter.project_name}}'
    var_1 = [var_0]
    var_2 = {}
    var_3 = '/output/project'
    var_4 = True
    var_5 = '/repo'
    var_6 = 'cookiecutter'
    var_7 = '_jinja2_env_vars'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = '/output'
    var_12 = False
    var_13 = module_0.generate_files(var_5, var_10, var_11, var_4, accept_hooks=var_12)
    assert var_13 == '/output/project'



