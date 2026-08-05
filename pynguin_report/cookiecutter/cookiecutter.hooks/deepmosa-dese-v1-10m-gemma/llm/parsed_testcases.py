####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/10 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 8/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project'
    var_1 = [var_0]
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/tmp/project'
    var_7 = module_0.run_hook(var_5, var_6, var_4)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project_1'
    var_1 = '/tmp/hooks/pre_gen_project_2'
    var_2 = [var_0, var_1]
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = module_0.run_hook(var_6, var_7, var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_command_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/9 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/6 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 3/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]
    var_5 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'script.sh'
    var_1 = module_0.run_script(var_0)
    var_2 = [var_0]
    var_3 = True
    var_4 = '.'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_no_matching_hooks. Retrieved 3/11 statements.
# Partially parsed test_find_hook_success. Retrieved 4/16 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = '#!/bin/bash'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post-commit'
    var_2 = f'{var_1}.sh'
    var_3 = '#!/bin/bash'
    assert var_3 == 1

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit'
    var_2 = f'{var_1}.sh~'
    var_3 = '#!/bin/bash'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_script_with_context_success. Retrieved 9/24 statements.
# Partially parsed test_run_script_with_context_extension_handling. Retrieved 7/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hello_{{ name }}'
    var_1 = 'name'
    var_2 = 'world'
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = 'template.py'
    var_6 = module_0.run_script_with_context(var_5, var_4, var_3)
    var_7 = b'hello_world'
    var_8 = '/tmp/test_script.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'data: {{ value }}'
    var_1 = 'value'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = 'script.txt'
    var_5 = '.'
    var_6 = module_0.run_script_with_context(var_4, var_5, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_original_dir_when_no_hook_exists. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_returns_tmp_dir_when_hook_exists. Retrieved 7/16 statements.
# Partially parsed test_run_pre_prompt_hook_raises_exception_on_script_failure. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'no_hooks_repo'

def test_case_0():
    var_0 = 'with_hooks_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = "print('hello')"
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = None
    var_6 = lambda script, cwd: var_5

def test_case_0():
    var_0 = 'fail_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'exit(1)'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = "raise FailedHookException('fail')"
    var_6 = exec(var_5)
    var_7 = lambda script, cwd: var_6



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_hook_type_hint_validation. Retrieved 5/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_directory_for_type_check'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = None
    var_4 = type(var_3)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = '/path/to/pre-commit'
    var_5 = module_0.valid_hook(var_4, var_0)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit'
    var_3 = 'wrong-name'
    var_4 = module_0.valid_hook(var_2, var_3)
    assert var_4 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/unknown'
    var_3 = 'unknown'
    var_4 = module_0.valid_hook(var_2, var_3)
    assert var_4 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit~'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/other'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post-merge'
    var_1 = [var_0]
    var_2 = 'C:\\Users\\Admin\\hooks\\post-merge.py'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'non_existent_hook'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/13 statements.
# Partially parsed test_run_script_shell_script_windows. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 2/10 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 2/7 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 2/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]
    var_5 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    assert var_0 == 'Hook script failed (exit status: 1)'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    assert var_0 == 'Hook script failed, might be an empty file or missing a shebang'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #11
#--------------------------

# Failed to parse test_find_hook_signature_type_hinting.




# Parsed testcases at query #12
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys; sys.exit(0)'
    var_3 = 'pre_prompt'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/17 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_deletes_project. Retrieved 10/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_does_not_delete_project. Retrieved 9/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/repo'
    var_1 = 'post_gen_project'
    var_2 = '/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Failed'
    var_1 = '/repo'
    var_2 = 'post_gen_project'
    var_3 = '/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = '/project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Failed'
    var_1 = '/repo'
    var_2 = 'post_gen_project'
    var_3 = '/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = 'pre-commit'
    var_5 = module_0.valid_hook(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'empty_repo'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/10 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 11/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project'
    var_4 = '/tmp/project'
    var_5 = module_0.run_hook(var_3, var_4, var_2)
    var_6 = False
    var_7 = '/tmp/post_gen_project.py'
    var_8 = '/tmp/project/hooks/pre_gen_project.py'
    var_9 = var_7 if var_6 else var_8
    var_10 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/project/hooks/pre_gen_project.py'
    var_1 = '/tmp/project/hooks/pre_gen_project.sh'
    var_2 = [var_0, var_1]
    var_3 = 'pre_gen_project'
    var_4 = '/tmp/project'
    var_5 = {}
    var_6 = module_0.run_hook(var_3, var_4, var_5)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = 'pre-commit'
    var_5 = module_0.valid_hook(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_hook_type_hint_validity. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'return'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/16 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/14 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/11 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 4/12 statements.
# Partially parsed test_run_script_os_error_generic. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = OSError(var_0)
    var_2 = 'test_script.py'
    var_3 = module_0.run_script(var_2)
    var_4 = str(var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec. Retrieved 4/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec. Retrieved 4/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/17 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_deletes_project. Retrieved 10/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_does_not_delete_project. Retrieved 9/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Failed'
    var_1 = '/tmp/repo'
    var_2 = 'post_gen_project'
    var_3 = '/tmp/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = '/tmp/project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Failed'
    var_1 = '/tmp/repo'
    var_2 = 'post_gen_project'
    var_3 = '/tmp/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_raises_enoexec_error. Retrieved 4/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = module_0.valid_hook(var_3, var_0)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/other-hook'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/unknown'
    var_3 = 'unknown'
    var_4 = module_0.valid_hook(var_2, var_3)
    assert var_4 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit~'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit.py'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = ''
    var_3 = module_0.valid_hook(var_2, var_2)
    assert var_3 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_script_with_context_success. Retrieved 18/38 statements.
# Partially parsed test_run_script_with_context_fails_on_render. Retrieved 15/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter.hooks.Path.read_text'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter.hooks.create_env_with_context'
    var_3 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'os.path.splitext'
    var_6 = 'script'
    var_7 = '.sh'
    var_8 = (var_6, var_7)
    var_9 = 'cookiejack.hooks.tempfile.NamedTemporaryFile'
    var_10 = 'script.sh'
    var_11 = '.'
    var_12 = 'name'
    var_13 = 'World'
    var_14 = {var_12: var_13}
    var_15 = module_0.run_script_with_context(var_10, var_11, var_14)
    var_16 = b'Hello World'
    var_17 = '/tmp/temp_script.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter.hooks.Path.read_text'
    var_1 = 'Error context'
    var_2 = 'cookiecutter.hooks.create_env_with_context'
    var_3 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_4 = 'os.path.splitext'
    var_5 = 'script'
    var_6 = '.sh'
    var_7 = (var_5, var_6)
    var_8 = 'Missing key'
    var_9 = 'script.sh'
    var_10 = '.'
    var_11 = {}
    var_12 = module_0.run_script_with_context(var_9, var_10, var_11)
    var_13 = 'KeyError should have been raised'
    var_14 = AssertionError(var_13)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_returns_none_when_directory_is_empty. Retrieved 1/5 statements.
# Partially parsed test_find_hook_returns_path_when_valid_hook_exists. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/13 statements.
# Partially parsed test_find_hook_ignores_mismatched_hook_name. Retrieved 2/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '#!/bin/bash\nexit 0'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'valid'
    var_1 = 'backup'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'content'
    var_1 = 'pre-commit'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_script_raises_enoexec_error. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_script_with_context_tempfile_suffix_matches_extension. Retrieved 11/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp/cwd'
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_8 = False
    var_9 = 'wb'
    var_10 = '.sh'



# Parsed testcases at query #11
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = 'pre-commit'
    var_5 = module_0.valid_hook(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir. Retrieved 3/25 statements.
# Partially parsed test_run_pre_prompt_hook_raises_exception_on_failed_script. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/bash\nexit 0'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = 'FailedHookException was not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_success. Retrieved 3/10 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'test_hook'

def test_case_0():
    var_0 = 'valid_hook_name'
    var_1 = f'{var_0}.sh'
    var_2 = "#!/bin/bash\necho 'hello'"

def test_case_0():
    var_0 = 'valid_hook_name'
    var_1 = f'{var_0}.sh~'
    var_2 = 'content'

def test_case_0():
    var_0 = 'valid_hook_name'
    var_1 = f'{var_0}.py'
    var_2 = "print('test')"
    var_3 = 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_success_windows. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/test.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]
    var_5 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/test.sh'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = module_0.valid_hook(var_3, var_0)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/post-checkout'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported'
    var_1 = [var_0]
    var_2 = 'pre-commit'
    var_3 = [var_2]
    var_4 = '/path/to/unsupported'
    var_5 = module_0.valid_hook(var_4, var_0)
    assert var_5 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit~'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = 'subdir/pre-commit'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit.sh'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_no_matching_hooks. Retrieved 3/11 statements.
# Partially parsed test_find_hook_success. Retrieved 4/16 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_directory_path_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = "#!/bin/bash\necho 'hello'"
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit'
    var_2 = f'{var_1}.sh'
    var_3 = "#!/bin/bash\necho 'hello'"
    assert var_3 == 1

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit'
    var_2 = f'{var_1}.sh~'
    var_3 = 'backup'



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hooks_returns_tmp_copy. Retrieved 8/28 statements.
# Partially parsed test_run_pre_prompt_hook_fails_on_bad_script. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493
    var_4 = 'hooks'
    var_5 = var_2 / var_4
    var_6 = 'pre_prompt.py'
    var_7 = var_5 / var_6

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = 493
    var_4 = 'Should have raised FailedHookException'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_windows. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/10 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/10 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/test.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]
    var_5 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/test.sh'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/tmp/test.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = OSError(var_0)
    var_2 = '/tmp/test.sh'
    var_3 = module_0.run_script(var_2)
    var_4 = str(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/10 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 11/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = [var_0]
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/tmp/project'
    var_7 = module_0.run_hook(var_5, var_6, var_4)
    var_8 = 0
    var_9 = var_1[var_8]
    var_10 = 'Running hook %s'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_script_exit_status_success. Retrieved 3/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_find_hook_signature_type_hints.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/10 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/10 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 4/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]
    var_5 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 'Hello {{ name }}!'
    var_1 = 'test_script.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'cookiecutter'
    var_5 = 'World'
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 7/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'some'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 14/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = 'os.listdir'
    var_3 = 'test_hook.py'
    var_4 = [var_3]
    var_5 = 'os.path.abspath'
    var_6 = lambda x: x
    var_7 = 'os.path.join'
    var_8 = lambda x, y: f'{x}/{y}'
    var_9 = '__main__.valid_hook'
    var_10 = False
    var_11 = 'non_existent_hook'
    var_12 = 'hooks'
    var_13 = module_0.find_hook(var_11, var_12)
    assert var_13 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_script_success_status. Retrieved 2/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_raises_on_failed_hook. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_hook.py'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'Hook failed'
    var_8 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #30
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'commit-msg'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = 'pre-commit'
    var_5 = module_0.valid_hook(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_find_hook_type_hint_validity. Retrieved 7/12 statements.


def test_case_0():
    var_0 = '/abs/path/to/hook1.py'
    var_1 = '/abs/path/to/hook2.py'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = None
    var_5 = type(var_4)
    var_6 = isinstance(var_3, var_5)



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_executes_and_returns_new_dir. Retrieved 4/20 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script_raises_exception. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = f'{var_1}.py'
    var_3 = f"import sys; print('executing'); sys.exit(0)"

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = f'{var_1}.py'
    var_3 = f'import sys; sys.exit(1)'
    var_4 = 'FailedHookException was not raised'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/10 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 11/18 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 13/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = [var_0]
    var_2 = 'some_var'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '/tmp/project'
    var_6 = 'pre_gen_project'
    var_7 = module_0.run_hook(var_6, var_5, var_4)
    var_8 = 0
    var_9 = var_1[var_8]
    var_10 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = '/tmp/hooks/pre_gen_project.py'
    var_2 = [var_0, var_1]
    var_3 = 'some_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '/tmp/project'
    var_7 = 'pre_gen_project'
    var_8 = module_0.run_hook(var_7, var_6, var_5)
    var_9 = 0
    var_10 = var_2[var_9]
    var_11 = 1
    var_12 = var_2[var_11]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_find_hook_returns_none_when_directory_is_empty. Retrieved 1/5 statements.
# Partially parsed test_find_hook_returns_path_when_valid_hook_exists. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_ignores_mismatched_hook_names. Retrieved 4/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'test_hook'

def test_case_0():
    var_0 = 'valid_name'
    var_1 = f'{var_0}.sh'
    var_2 = '#!/bin/bash\n'

def test_case_0():
    var_0 = 'valid_name'
    var_1 = f'{var_0}.sh~'
    var_2 = 'backup'

def test_case_0():
    var_0 = 'wrong_name'
    var_1 = f'{var_0}.sh'
    var_2 = 'content'
    var_3 = 'valid_name'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 7/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'some'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'os.listdir'
    var_4 = 'not_a_valid_hook.py'
    var_5 = [var_4]
    var_6 = lambda x: var_5
    var_7 = 'valid_hook'
    var_8 = False
    var_9 = lambda file, name: var_8
    var_10 = 'test_hook'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_find_hook_type_hint_validity. Retrieved 6/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks'
    var_1 = True
    var_2 = 'test_script.py'
    var_3 = '# dummy content'
    var_4 = 'test_script'
    var_5 = module_0.find_hook(var_4, var_3)



