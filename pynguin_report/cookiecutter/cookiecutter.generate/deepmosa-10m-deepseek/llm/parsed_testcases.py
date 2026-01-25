####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__run_hook_from_repo_dir_deprecation_warning. Retrieved 12/24 statements.
# Partially parsed test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir. Retrieved 8/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'test_hook'
    var_2 = '/path/to/project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'always'
    var_8 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_9 = -1
    var_10 = -1
    var_11 = '_run_hook_from_repo_dir'
    var_12 = -1
    var_13 = 'run_hook_from_repo_dir'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'test_hook'
    var_2 = '/path/to/project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 8/9 statements.
# Partially parsed test_render_and_create_dir_raises_output_dir_exists_exception. Retrieved 9/12 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 9/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = module_1.render_and_create_dir(var_5, var_2, var_3, var_4)
    var_7 = '/tmp/test_dir'
    var_8 = [var_7]
    var_9 = var_6[0]
    var_10 = var_6[1]
    assert var_10 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = module_1.render_and_create_dir(var_5, var_2, var_3, var_4)
    var_7 = bool(False)
    assert var_7 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = '/tmp/test_dir'
    var_7 = [var_6]
    var_8 = True
    var_9 = module_1.render_and_create_dir(var_5, var_2, var_3, var_4)
    var_10 = bool(False)
    assert var_10 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = '/tmp/test_dir'
    var_7 = [var_6]
    var_8 = True
    var_9 = module_1.render_and_create_dir(var_5, var_2, var_3, var_4, var_8)
    var_10 = [var_6]
    var_11 = var_9[0]
    var_12 = var_9[1]
    assert var_12 is False



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'new_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)
    var_5 = bool(var_0 == {})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = bool(var_2 == {'nested': {}, 'new_var': 'value'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['b', 'a']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6, var_1]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'var': 'new'})
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_generate_files_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_empty_context. Retrieved 4/5 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_keep_project_on_failure. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_copy_only_path. Retrieved 11/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'MyProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = {}
    var_2 = 'path/to/output'
    var_3 = module_0.generate_files(var_0, var_1, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'MyProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'MyProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'MyProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'MyProject'
    var_5 = 'path/to/copy'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'path/to/output'
    var_10 = module_0.generate_files(var_0, var_8, var_9)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'new_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)
    var_5 = bool(var_0 == {})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'dict_var': {'key1': 'value1', 'key2': 'value2'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = bool(var_4 == {'list_var': ['choice2', 'choice1']})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid_choice'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = 'choice3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoice_var': ['choice2', 'choice3']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'choice3'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'simple_var'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'simple_var': 'new_value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'level1'
    var_2 = 'level2'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value'
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_6, var_10)
    var_12 = bool(var_6 == {'nested': {'level1': {'level2': 'new_value'}}})
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid_value'
    var_2 = var_0.process_response(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'non_dict_value'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = var_4['key']
    assert var_8 == 'non_dict_value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 2/3 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = 'key'
    var_6 = bool('key' in var_4['cookiecutter'])
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = 'key'
    var_6 = bool('key' in var_4['cookiecutter'])
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 5/7 statements.
# Partially parsed test_render_and_create_dir_raises_error_on_existing_directory. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ name }}'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = ''
    var_5 = '/tmp'
    var_6 = module_1.render_and_create_dir(var_4, var_2, var_5, var_3)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ name }}'
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ name }}'
    var_5 = True



# Parsed testcases at query #10
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'true'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 't'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'yes'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 'y'
    var_10 = var_0.process_response(var_9)
    assert var_10 is True
    var_11 = 'on'
    var_12 = var_0.process_response(var_11)
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/9 statements.
# Partially parsed test_generate_files_overwrites_existing_directory. Retrieved 9/12 statements.
# Partially parsed test_generate_files_skips_existing_files. Retrieved 9/10 statements.
# Partially parsed test_generate_files_accepts_hooks. Retrieved 9/10 statements.
# Partially parsed test_generate_files_keeps_project_on_failure. Retrieved 9/10 statements.
# Partially parsed test_generate_files_raises_exception_on_existing_dir. Retrieved 8/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_delete_project_on_failure_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/6 statements.
# Partially parsed test_render_and_create_dir_raises_output_dir_exists_exception. Retrieved 6/11 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 5/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2]
    var_5 = True
    var_6 = [var_2, var_0]
    var_7 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2]
    var_5 = True
    var_6 = [var_2, var_0]



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = '*.json'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = '*.json'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'file.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'file.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_empty_dirname_raises_exception. Retrieved 4/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = ''



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = {}
    var_2 = 'output'
    var_3 = False
    var_4 = False
    var_5 = True
    var_6 = True
    var_7 = module_0.generate_files(var_0, var_1, var_2, var_3, var_4, var_5, var_6)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'new_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)
    var_5 = bool(var_0 == {})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = bool(var_2 == {'dict_var': {'new_var': 'value'}})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoice_var': ['a', 'b']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
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
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice_var': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'dict_var': {'key1': 'value1', 'key2': 'new_value2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'var': 'new_value'})
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #20
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid_choice'
    var_2 = var_0.process_response(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = False
    var_2 = module_0.generate_files(var_0, accept_hooks=var_1)
    var_3 = 'project_dir'
    var_4 = bool('project_dir' in var_2)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'true'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 't'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'yes'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 'y'
    var_10 = var_0.process_response(var_9)
    assert var_10 is True
    var_11 = 'on'
    var_12 = var_0.process_response(var_11)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '0'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'false'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = 'f'
    var_6 = var_0.process_response(var_5)
    assert var_6 is False
    var_7 = 'no'
    var_8 = var_0.process_response(var_7)
    assert var_8 is False
    var_9 = 'n'
    var_10 = var_0.process_response(var_9)
    assert var_10 is False
    var_11 = 'off'
    var_12 = var_0.process_response(var_11)
    assert var_12 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = var_0.process_response(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 2/15 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/11 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/18 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/18 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 7/20 statements.
# Partially parsed test_generate_context_with_nested_dictionaries. Retrieved 7/20 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = '{"key": "original"}'
    var_1 = 'key'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 0

def test_case_0():
    var_0 = '{"key": "original"}'
    var_1 = 'key'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = 0

def test_case_0():
    var_0 = '{"key": "original"}'
    var_1 = 'key'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = 0

def test_case_0():
    var_0 = '{"nested": {"key": "original"}}'
    var_1 = 'nested'
    var_2 = 'key'
    var_3 = 'new'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 8/17 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 10/19 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 10/19 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/11 statements.
# Partially parsed test_generate_context_with_boolean_overwrite. Retrieved 6/15 statements.
# Partially parsed test_generate_context_with_list_overwrite. Retrieved 9/18 statements.
# Partially parsed test_generate_context_with_dict_overwrite. Retrieved 11/20 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.generate_context(var_2)
    var_8 = bool(var_7 == {'cookiecutter': var_6})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'new_value'
    var_8 = {var_0: var_7}
    var_9 = module_0.generate_context(var_2, var_8)
    var_10 = var_9['cookiecutter']['key1']
    assert var_10 == 'new_value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'extra_value'
    var_8 = {var_0: var_7}
    var_9 = module_0.generate_context(var_2, extra_context=var_8)
    var_10 = var_9['cookiecutter']['key1']
    assert var_10 == 'extra_value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, extra_context=var_4)
    var_6 = var_5['cookiecutter']['key1']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.generate_context(var_2, extra_context=var_7)
    var_9 = var_8['cookiecutter']['key1']
    var_10 = bool(var_8['cookiecutter']['key1'] == ['a', 'b'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'subkey1'
    var_2 = 'subkey2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.generate_context(var_2, extra_context=var_9)
    var_11 = var_10['cookiecutter']['key1']['subkey1']
    assert var_11 == 'new_value'
    var_12 = var_10['cookiecutter']['key1']['subkey2']
    assert var_12 == 'value2'



# Parsed testcases at query #27
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = True
    var_8 = False
    var_9 = True
    var_10 = False
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = False
    var_8 = False
    var_9 = True
    var_10 = False
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = False
    var_8 = True
    var_9 = True
    var_10 = False
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = False
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = False
    var_8 = False
    var_9 = True
    var_10 = True
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_delete_project_on_failure_false_when_output_directory_not_created. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_keep_project_on_failure_true. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_both_conditions_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 9/21 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/11 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 13/25 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 11/23 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 14/26 statements.
# Partially parsed test_generate_context_with_boolean_conversion. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = '.'
    var_5 = tmp.name.split(var_4)[var_3]
    var_6 = {var_0: var_1}
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_8]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_1}
    var_6 = 0
    var_7 = '.'
    var_8 = tmp.name.split(var_7)[var_6]
    var_9 = [var_1, var_2]
    var_10 = {var_0: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_11]
    var_13 = [var_12]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = '.'
    var_7 = tmp.name.split(var_6)[var_5]
    var_8 = {var_0: var_3}
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_10]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_2}
    var_7 = 0
    var_8 = '.'
    var_9 = tmp.name.split(var_8)[var_7]
    var_10 = [var_2, var_1]
    var_11 = {var_0: var_10}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = '.'
    var_7 = tmp.name.split(var_6)[var_5]
    var_8 = {var_0: var_1}
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_10]



# Parsed testcases at query #30
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'output_dir'
    var_5 = True
    var_6 = module_0.generate_files(var_0, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_files_with_valid_input. Retrieved 8/9 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 9/11 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 9/10 statements.
# Partially parsed test_generate_files_without_accept_hooks. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_keep_project_on_failure. Retrieved 9/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = False
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/invalid/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = {}
    var_2 = '/path/to/output'
    var_3 = module_0.generate_files(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_delete_project_on_failure_false_when_output_directory_not_created. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_keep_project_on_failure_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #33
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'non_existent_file.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_true. Retrieved 9/13 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.create_env_with_context(var_6)
    var_8 = '.'
    var_9 = any(var_2)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #35
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = {}
    var_2 = None
    var_3 = module_0.generate_context(var_0, var_1, var_2)
    var_4 = 'test'
    var_5 = bool('test' in var_3)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 3/9 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/11 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/11 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 8/19 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, var_5)
    var_7 = 0
    var_8 = 'Invalid default received'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = 'extra_value'
    var_6 = {var_0: var_5}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_59_evaluates_to_false. Retrieved 11/13 statements.


import cookiecutter.utils as module_0
import cookiecutter.find as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.create_env_with_context(var_4)
    var_6 = '/path/to/repo'
    var_7 = module_1.find_template(var_6, var_5)
    var_8 = '/path/to/project'
    var_9 = False
    var_10 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_undefined_error_raises_undefined_variable_in_template. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'cookiecutter'
    var_3 = 'invalid_var'
    var_4 = '{{ undefined_var }}'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'output'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 9/15 statements.
# Partially parsed test_generate_file_binary_file. Retrieved 9/15 statements.
# Partially parsed test_generate_file_text_file. Retrieved 9/16 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 8/10 statements.
# Partially parsed test_generate_file_new_lines_config. Retrieved 11/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'binary content'
    assert var_7 == b'binary content'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello {{ name }}'
    assert var_7 == 'Hello World'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Hello {{ name }}'
    assert var_9 == 'Hello World'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generate_file_creates_file_with_correct_content. Retrieved 9/11 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 11/16 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 10/14 statements.
# Partially parsed test_generate_file_uses_custom_newline. Retrieved 11/13 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 13/20 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'existing content'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = b'\x00\x01\x02\x03'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = '_new_lines'
    var_5 = 'value'
    var_6 = '\r\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.Environment()
    var_10 = module_1.generate_file(var_0, var_1, var_8, var_9)
    var_11 = '\r\n'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'content'
    var_9 = 420
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'template.txt'
    var_12 = 511



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 9/15 statements.
# Partially parsed test_generate_file_binary_file. Retrieved 8/14 statements.
# Partially parsed test_generate_file_text_file. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 6/11 statements.
# Partially parsed test_generate_file_new_lines. Retrieved 14/20 statements.
# Partially parsed test_generate_file_template_syntax_error. Retrieved 9/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = True
    var_6 = 'file.txt'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_4)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'binary content'
    assert var_5 == b'binary content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = 'binary.bin'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello {{ cookiecutter.variable }}'
    assert var_11 == 'Hello value'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = 'file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello\nWorld'
    assert var_11 == 'Hello\nWorld'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = 'file.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = {}
    var_3 = '/tmp/template'
    var_4 = module_0.FileSystemLoader(var_3)
    var_5 = module_1.Environment(loader=var_4)
    var_6 = True
    var_7 = 'Hello {{ invalid_syntax }'
    var_8 = module_2.generate_file(var_0, var_1, var_2, var_5)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 6/10 statements.
# Partially parsed test_generate_file_text_file. Retrieved 8/17 statements.
# Partially parsed test_generate_file_skip_existing. Retrieved 8/16 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 5/7 statements.
# Partially parsed test_generate_file_newlines. Retrieved 10/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_binary.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'\x00\x01\x02\x03'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ name }}'
    var_6 = {var_1: var_5}
    var_7 = 'Hello {{ name }}'
    assert var_7 == 'Hello Test'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_skip.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'content'
    var_5 = True
    var_6 = 'existing content'
    assert var_6 == 'existing content'
    var_7 = module_1.generate_file(var_0, var_1, var_2, var_3, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_newlines.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'line1\nline2'
    var_8 = {var_1: var_7}
    var_9 = 'line1\nline2'
    var_10 = b'\r\n'
    var_11 = bool(b'\r\n' in var_9)
    assert var_11 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 6/10 statements.
# Partially parsed test_generate_file_text_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 8/17 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 5/7 statements.
# Partially parsed test_generate_file_new_lines. Retrieved 10/16 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/binary_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'\x00\x01\x02\x03'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello {{ cookiecutter.key }}'
    assert var_8 == 'Hello value'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/skip_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'content'
    var_5 = True
    var_6 = 'existing content'
    assert var_6 == 'existing content'
    var_7 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Line1\nLine2'
    assert var_8 == 'Line1\r\nLine2'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_new_lines_from_context. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_False. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'new'
    var_7 = bool('new' not in var_2)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = var_4['nested']['new']
    assert var_10 == 'value'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['choices']
    var_10 = bool(var_5['choices'] == ['b', 'a'])
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
    var_7 = [var_6, var_1]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = 'd'

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = 'd'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = var_6['nested']['b']
    assert var_11 == 3

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['flag']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'invalid'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['var']
    assert var_6 == 'new'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 2/3 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/6 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/6 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 8/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/valid_context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'valid_context'
    var_3 = bool('valid_context' in var_1)
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/invalid_context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test_data/valid_context.json'
    var_4 = module_0.generate_context(var_3, var_2)
    var_5 = 'valid_context'
    var_6 = bool('valid_context' in var_4)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key2'
    var_1 = 'value2'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test_data/valid_context.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)
    var_5 = 'valid_context'
    var_6 = bool('valid_context' in var_4)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'tests/test_data/valid_context.json'
    var_7 = module_0.generate_context(var_6, var_2, var_5)
    var_8 = 'valid_context'
    var_9 = bool('valid_context' in var_7)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_empty_dirname. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 4/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_raises_error_when_dirname_is_empty. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_raises_error_when_directory_exists. Retrieved 8/13 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 8/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '{{ name }}_dir'
    var_7 = '/tmp/test_dir'
    var_8 = [var_7]

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = ''
    var_7 = bool(False)
    assert var_7 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '{{ name }}_dir'
    var_7 = '/tmp/test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '{{ name }}_dir'
    var_7 = '/tmp/test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = [var_7]



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = bool(var_4 == {'nested': {'existing': 'value', 'new': 'value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['b', 'a']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new'})
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generate_file_creates_output_file. Retrieved 5/24 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 7/27 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 5/23 statements.
# Partially parsed test_generate_file_handles_empty_filename. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = 'Hello {{ name }}'
    assert var_4 == 'Hello World'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.txt'
    var_4 = 'Hello {{ name }}'
    var_5 = 'Existing content'
    assert var_5 == 'Existing content'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.bin'
    var_4 = b'\x00\x01\x02\x03'
    assert var_4 == b'\x00\x01\x02\x03'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = ''
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{ name }}.txt'
    var_6 = 'content'
    var_7 = '.txt'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 5/7 statements.
# Partially parsed test_generate_file_text_file. Retrieved 9/11 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 10/12 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 9/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/empty_file_name.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/bad_template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 9/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = True
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/valid_context.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)
    var_8 = 'valid_context'
    var_9 = bool('valid_context' in var_7)
    assert var_9 is True
    var_10 = var_7['valid_context']['key1']
    assert var_10 == 'value1'
    var_11 = var_7['valid_context']['key2']
    assert var_11 == 'value2'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/invalid_context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/valid_context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'valid_context'
    var_3 = bool('valid_context' in var_1)
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/valid_context.json'
    var_1 = 'key1'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/valid_context.json'
    var_1 = 'key2'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_4, var_6, in_dictionary_variable=var_7)
    var_9 = bool(var_4 == {'existing': {'nested': 'value'}, 'new': 'value'})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['a', 'b']})
    assert var_9 is True

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
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new_value'})
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 10/15 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/example/project'
    var_1 = 'example.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = True
    var_8 = 'test content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_overwrite_if_exists. Retrieved 5/8 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = 'test_dir'
    var_5 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = b'\x00\x01\x02\x03'
    assert var_0 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'JSON decoding error while loading'



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_invalid_json_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": "test_project"}'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": "test_project"'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": "test_project"}'
    var_2 = 'name'
    var_3 = 'new_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": "test_project"}'
    var_2 = 'name'
    var_3 = 'extra_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": "test_project"}'
    var_2 = 'name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'extra_project'
    var_6 = {var_2: var_5}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"name": ["test_project"]}'
    var_2 = 'name'
    var_3 = 'invalid_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 7/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'test'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 3/10 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/12 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/12 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 7/14 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 7/14 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 5/12 statements.
# Partially parsed test_generate_context_with_invalid_boolean_variable. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = 'new_value'
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)



# Parsed testcases at query #19
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.doc'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*.doc'
    var_4 = 'templates/*.txt'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.doc'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*.txt'
    var_4 = 'templates/*.pdf'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is False



# Parsed testcases at query #20
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test__run_hook_from_repo_dir_emits_deprecation_warning. Retrieved 12/19 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    assert var_4 == 1
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_10 = 0
    var_11 = var_6.category
    var_12 = "_run_hook_from_repo_dir' function is deprecated"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_and_create_dir_successful_creation. Retrieved 5/18 statements.
# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 4/19 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 5/19 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing'
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing'
    var_4 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_new_lines_in_context. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/21 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 13/16 statements.
# Partially parsed test_generate_file_uses_newline_from_context. Retrieved 15/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello {{ cookiecutter.name }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello {{ cookiecutter.name }}'
    var_12 = 'Existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'Test'
    var_6 = '\r\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_files_with_copy_only_path. Retrieved 12/18 statements.
# Partially parsed test_generate_files_with_rendered_path. Retrieved 11/19 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 12/22 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 12/22 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 11/17 statements.
# Partially parsed test_generate_files_with_keep_project_on_failure. Retrieved 11/17 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = '/tmp/output'
    var_8 = True
    var_9 = 'test'
    var_10 = module_0.generate_files(var_0, var_6, var_7)
    var_11 = 'file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = '{{ cookiecutter.name }}'
    assert var_8 == 'project'
    var_9 = module_0.generate_files(var_0, var_5, var_6)
    var_10 = 'file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = '{{ cookiecutter.name }}'
    var_9 = 'old content'
    assert var_9 == 'project'
    var_10 = module_0.generate_files(var_0, var_5, var_6, var_7)
    var_11 = 'file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = '{{ cookiecutter.name }}'
    var_9 = 'old content'
    assert var_9 == 'old content'
    var_10 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)
    var_11 = 'file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = 'print("pre_gen_project")'
    var_9 = 'print("post_gen_project")'
    var_10 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = '{{ invalid_template }}'
    var_9 = True
    var_10 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_9)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_render_and_create_dir_overwrite_if_exists. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = True
    var_6 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_new_lines_in_context. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_render_and_create_dir_with_existing_dir_and_no_overwrite_raises_exception. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_and_overwrite_does_not_raise_exception. Retrieved 8/14 statements.
# Partially parsed test_render_and_create_dir_with_non_existing_dir_creates_dir. Retrieved 8/12 statements.
# Partially parsed test_render_and_create_dir_returns_correct_tuple. Retrieved 8/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = ''
    var_2 = {}
    var_3 = 'output_dir'
    var_4 = module_1.render_and_create_dir(var_1, var_2, var_3, var_0)
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'output_dir'
    var_5 = 'test_dir'
    var_6 = [var_4]
    var_7 = True
    var_8 = [var_4, var_5]
    var_9 = False
    var_10 = module_1.render_and_create_dir(var_5, var_3, var_4, var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'output_dir'
    var_5 = 'test_dir'
    var_6 = [var_4]
    var_7 = True
    var_8 = [var_4, var_5]
    var_9 = module_1.render_and_create_dir(var_5, var_3, var_4, var_0, var_7)
    var_10 = [var_4, var_5]

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'output_dir'
    var_5 = 'test_dir'
    var_6 = [var_4]
    var_7 = True
    var_8 = module_1.render_and_create_dir(var_5, var_3, var_4, var_0)
    var_9 = [var_4, var_5]

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'output_dir'
    var_5 = 'test_dir'
    var_6 = [var_4]
    var_7 = True
    var_8 = module_1.render_and_create_dir(var_5, var_3, var_4, var_0)
    var_9 = [var_4, var_5]



# Parsed testcases at query #30
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 6/10 statements.
# Partially parsed test_generate_file_text_render. Retrieved 9/14 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 7/14 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 5/7 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_5 = False

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_9 = bool(var_2 != var_3)
    assert var_9 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/text_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'existing content'
    assert var_5 == 'existing content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_4)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'tests/data/invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_8 = True
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 7/13 statements.
# Partially parsed test_generate_file_text_file. Retrieved 11/17 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 9/16 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 9/13 statements.
# Partially parsed test_generate_file_new_lines. Retrieved 10/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'\x00\x01\x02\x03'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)
    assert var_5 == b'\x00\x01\x02\x03'
    var_6 = 'binary_file'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello {{ cookiecutter.name }}'
    assert var_8 == 'Hello test'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_10 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'Hello'
    var_5 = 'text_file.txt'
    var_6 = 'Existing'
    assert var_6 == 'Existing'
    var_7 = True
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/empty_file'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello'
    assert var_8 == 'Hello\r\n'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 7/11 statements.
# Partially parsed test_generate_file_text_file. Retrieved 11/15 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 9/16 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 6/8 statements.
# Partially parsed test_generate_file_template_syntax_error. Retrieved 6/9 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'binary data'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_6 = 'binary_file'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello {{ name }}'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_10 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/skip_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'content'
    var_5 = 'skip_file.txt'
    var_6 = 'existing content'
    assert var_6 == 'existing content'
    var_7 = True
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_5 = ''

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/invalid_template.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'Hello {{ name'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_6 = bool(True)
    assert var_6 is True
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 6/8 statements.
# Partially parsed test_generate_file_text_file. Retrieved 8/10 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 9/11 statements.
# Partially parsed test_generate_file_with_new_lines. Retrieved 10/12 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 8/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_5 = 'binary_file.bin'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_7 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)
    var_8 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_9 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/empty_file'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_7 = ''

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_files_with_valid_input. Retrieved 8/9 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 9/12 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 9/12 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 9/10 statements.
# Partially parsed test_generate_files_with_keep_project_on_failure. Retrieved 9/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output_dir'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output_dir'
    var_7 = False
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)
    var_9 = bool(os.path.exists(os.path.join(var_6, 'TestProject')))
    assert var_9 is True



