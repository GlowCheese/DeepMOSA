####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit.py'
    var_4 = module_0.valid_hook(var_3, var_0)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/other.py'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/unknown.py'
    var_3 = 'unknown'
    var_4 = module_0.valid_hook(var_2, var_3)
    assert var_4 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit.py~'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0.valid_hook(var_0, var_0)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/9 statements.
# Partially parsed test_run_hook_executes_found_scripts. Retrieved 11/15 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 13/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/project/hooks/post_gen_project'
    var_1 = [var_0]
    var_2 = 'post_gen_project'
    var_3 = '/tmp/project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = module_0.run_hook(var_2, var_3, var_6)
    var_8 = 0
    var_9 = var_1[var_8]
    var_10 = {var_4: var_5}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/project/hooks/post_gen_project'
    var_1 = '/tmp/project/hooks/other_hook'
    var_2 = [var_0, var_1]
    var_3 = 'post_gen_project'
    var_4 = '/tmp/project'
    var_5 = {}
    var_6 = module_0.run_hook(var_3, var_4, var_5)
    var_7 = 0
    var_8 = var_2[var_7]
    var_9 = {}
    var_10 = 1
    var_11 = var_2[var_10]
    var_12 = {}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/12 statements.
# Partially parsed test_run_script_shell_script_success_windows. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True



# Parsed testcases at query #4
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
    var_0 = []
    var_1 = ''
    var_2 = module_0.valid_hook(var_1, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit.sh'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/11 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 8/16 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 13/20 statements.


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
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project'
    var_6 = module_0.run_hook(var_4, var_5, var_3)
    var_7 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = '/tmp/hooks/pre_gen_project.py'
    var_2 = [var_0, var_1]
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = module_0.run_hook(var_6, var_7, var_5)
    var_9 = 0
    var_10 = var_2[var_9]
    var_11 = 1
    var_12 = var_2[var_11]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_valid_script_found. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_invalid_names. Retrieved 2/7 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 2/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'test_hook'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh'
    var_2 = '#!/bin/sh'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'wrong_name.sh'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh~'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_hook_type_hint_is_correct. Retrieved 5/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'hook_name'
    var_4 = 'hooks_dir'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = ''
    var_2 = 'non_existent_hook'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks.
# Partially parsed test_run_pre_prompt_hook_with_valid_hooks. Retrieved 3/23 statements.
# Partially parsed test_run_pre_prompt_hook_failure. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys; sys.exit(0)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys; sys.exit(1)'
    var_3 = module_0.run_pre_prompt_hook(var_2)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_early_when_no_scripts_found.




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 4/9 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 13/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/project/hooks/pre_gen_project_1.sh'
    var_1 = '/tmp/project/hooks/pre_gen_project_2.sh'
    var_2 = [var_0, var_1]
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = module_0.run_hook(var_6, var_7, var_5)
    var_9 = 0
    var_10 = var_2[var_9]
    var_11 = 1
    var_12 = var_2[var_11]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks_in_dir. Retrieved 2/8 statements.
# Partially parsed test_find_hook_returns_correct_path_for_valid_hook. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'some_hook'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = ''
    var_1 = 'target_hook'

def test_case_0():
    var_0 = 'target_hook.py'
    var_1 = ''
    var_2 = 'target_hook'

def test_case_0():
    var_0 = 'target_hook.py~'
    var_1 = ''
    var_2 = 'target_hook'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 2/7 statements.


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
    var_4 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 13/22 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'not_a_hook.txt'
    var_2 = 'content'
    var_3 = 'os.path.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'os.listdir'
    var_7 = [var_1]
    var_8 = lambda x: var_7
    var_9 = 'valid_hook'
    var_10 = False
    var_11 = lambda file, name: var_10
    var_12 = 'some_hook'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir. Retrieved 3/20 statements.
# Partially parsed test_run_pre_prompt_hook_fails_on_bad_script. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = f"import sys; print('running'); sys.exit(0)"
    var_3 = bool(var_0)
    assert var_3 is True
    var_4 = 'cookiecutter'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys; sys.exit(1)'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/9 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/9 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 3/10 statements.


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
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed (error: [Errno 13] Permission denied)'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_type_signature_valid. Retrieved 5/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'hook_name'
    var_4 = 'hooks_dir'



# Parsed testcases at query #9
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-merge'
    var_2 = [var_0, var_1]
    var_3 = [var_0]
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
    var_2 = '/path/to/unknown-hook'
    var_3 = 'unknown-hook'
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
    var_0 = []
    var_1 = ''
    var_2 = module_0.valid_hook(var_1, var_1)
    assert var_2 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 7/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'some'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/15 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/14 statements.
# Partially parsed test_run_script_windows_shell_true. Retrieved 6/14 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/11 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/12 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 5/12 statements.


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
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'C:\\scripts\\test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'C:\\Python\\python.exe'
    var_3 = [var_2, var_0]
    var_4 = True
    var_5 = '.'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'No such file or directory'
    var_2 = OSError(var_0, var_1)
    var_3 = '/path/to/nonexistent.py'
    var_4 = module_0.run_script(var_3)
    var_5 = 'Hook script failed (error: [Errno 2] No such file or directory)'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_valid_hook_found. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_ignores_mismatched_name. Retrieved 3/10 statements.


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
    var_1 = '# dummy content'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit~'
    var_1 = '# dummy content'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'post-commit'
    var_1 = '# dummy content'
    var_2 = 'pre-commit'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 8/15 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_deletes_project. Retrieved 9/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_does_not_delete_project. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = 'Failed'
    var_7 = [var_6]
    var_8 = {}
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_9)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'post_gen_project'
    var_2 = '/template/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = 'Failed'
    var_7 = [var_6]
    var_8 = {}
    var_9 = False
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_9)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks_found. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = ''
    var_2 = 'target_hook'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_hook_type_hint_validation. Retrieved 6/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = None
    var_4 = lambda : var_3
    var_5 = type(var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 5/12 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 4/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/9 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/9 statements.
# Partially parsed test_run_script_oserror_generic. Retrieved 4/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed (error: '
    var_5 = bool('Hook script failed (error: ' in var_3)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-merge'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/pre-commit'
    var_4 = module_0.valid_hook(var_3, var_0)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit'
    var_3 = 'post-merge'
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
    var_2 = '/path/to/pre-commit.sh'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = []
    var_1 = ''
    var_2 = module_0.valid_hook(var_1, var_1)
    assert var_2 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 6/11 statements.
# Partially parsed test_run_hook_executes_found_scripts. Retrieved 7/14 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 13/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/project'
    var_5 = 'pre_gen_project'
    var_6 = module_0.run_hook(var_5, var_4, var_3)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = '/tmp/hooks/pre_gen_project.py'
    var_2 = [var_0, var_1]
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = '/tmp/project'
    var_7 = 'pre_gen_project'
    var_8 = module_0.run_hook(var_7, var_6, var_5)
    var_9 = 0
    var_10 = var_2[var_9]
    var_11 = 1
    var_12 = var_2[var_11]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 8/16 statements.


import pathlib as module_0
import cookiecutter.hooks as module_1

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = 'some'
    var_6 = 'context'
    var_7 = {var_5: var_6}
    var_8 = module_1.run_hook(var_0, var_4, var_7)
    var_9 = 'No %s hook found'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 8/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_deletes_project. Retrieved 10/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_does_not_delete_project. Retrieved 9/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/repo'
    var_4 = 'post_gen_project'
    var_5 = '/tmp/project'
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_3, var_4, var_5, var_2, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'Failed'
    var_4 = [var_3]
    var_5 = {}
    var_6 = '/tmp/repo'
    var_7 = 'post_gen_project'
    var_8 = '/tmp/project'
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_6, var_7, var_8, var_2, var_9)
    var_11 = '/tmp/project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'Failed'
    var_4 = [var_3]
    var_5 = {}
    var_6 = '/tmp/repo'
    var_7 = 'post_gen_project'
    var_8 = '/tmp/project'
    var_9 = False
    var_10 = module_0.run_hook_from_repo_dir(var_6, var_7, var_8, var_2, var_9)



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_new_tmp_dir. Retrieved 7/26 statements.
# Partially parsed test_run_pre_prompt_hook_raises_failed_hook_exception_on_script_failure. Retrieved 7/26 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = None
    var_4 = [var_0]
    var_5 = '/tmp/fake_repo'
    var_6 = [var_5]
    var_7 = {}
    var_8 = [var_5]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = 'exit 1'
    var_3 = None
    var_4 = [var_0]
    var_5 = '/tmp/fake_repo'
    var_6 = [var_5]
    var_7 = {}
    var_8 = 'Original error'
    var_9 = [var_8]
    var_10 = {}
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_script_raises_enoexec_exception. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_valid_hook_found. Retrieved 11/25 statements.
# Partially parsed test_find_hook_no_matching_hooks. Retrieved 11/19 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 11/19 statements.
# Partially parsed test_find_hook_empty_directory. Retrieved 8/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'your_module._HOOKS'
    var_2 = 'pre-commit'
    var_3 = 'post-checkout'
    var_4 = [var_2, var_3]
    var_5 = 'pre-commit.sh'
    var_6 = '#!/bin/bash\nexit 0'
    var_7 = 'os.listdir'
    var_8 = [var_5]
    var_9 = lambda path: var_8
    var_10 = 'os.path.isdir'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'your_module._HOOKS'
    var_2 = 'pre-commit'
    var_3 = [var_2]
    var_4 = 'os.listdir'
    var_5 = 'wrong-name.sh'
    var_6 = [var_5]
    var_7 = lambda path: var_6
    var_8 = 'os.path.isdir'
    var_9 = True
    var_10 = lambda path: var_9

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'your_module._HOOKS'
    var_2 = 'pre-commit'
    var_3 = [var_2]
    var_4 = 'os.listdir'
    var_5 = 'pre-commit.sh~'
    var_6 = [var_5]
    var_7 = lambda path: var_6
    var_8 = 'os.path.isdir'
    var_9 = True
    var_10 = lambda path: var_9

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'os.listdir'
    var_2 = []
    var_3 = lambda path: var_2
    var_4 = 'os.path.isdir'
    var_5 = True
    var_6 = lambda path: var_5
    var_7 = 'pre-commit'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hooks_returns_original_dir. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_runs_and_returns_tmp_dir. Retrieved 5/22 statements.
# Partially parsed test_run_pre_prompt_hook_raises_exception_on_script_failure. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'no_hooks_repo'

def test_case_0():
    var_0 = 'valid_hooks_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'
    var_4 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'failing_hooks_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsys_exit(1)'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'FailedHookException was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 7/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_signature_validates. Retrieved 5/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = None
    var_4 = type(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_hook_success. Retrieved 6/19 statements.
# Partially parsed test_find_hook_no_matching_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_multiple_valid_files. Retrieved 6/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'pre-commit'
    var_4 = 'post-commit'
    var_5 = 'pre-commit'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'echo 1'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'echo 1'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit'
    var_2 = 'pre-commit.sh'
    var_3 = 'pre-commit.py'
    var_4 = ''
    var_5 = ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_script_raises_enoexec_exception. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_valid_script_found. Retrieved 3/12 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/9 statements.
# Partially parsed test_find_hook_ignores_mismatched_name. Retrieved 3/9 statements.


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
    var_0 = 'pre-commit~'
    var_1 = '#!/bin/bash\nexit 0'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '#!/bin/bash\nexit 0'
    var_2 = 'post-commit'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_signature_validity. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'hook_name'
    var_1 = 'hook_name'
    var_2 = 'hooks_dir'



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 11/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = []
    var_2 = 'cookiecutter.hooks.logger'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project'
    var_6 = 'some'
    var_7 = 'context'
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No %s hook found'



