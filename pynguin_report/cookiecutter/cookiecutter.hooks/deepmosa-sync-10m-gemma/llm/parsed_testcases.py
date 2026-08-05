####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_script_with_context_success. Retrieved 10/27 statements.
# Partially parsed test_run_script_with_context_failure_on_render. Retrieved 9/20 statements.
# Partially parsed test_run_script_with_context_handles_different_extensions. Retrieved 6/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/script'
    var_1 = '.py'
    var_2 = 'name'
    var_3 = 'world'
    var_4 = {var_2: var_3}
    var_5 = '/tmp/script.py'
    var_6 = '.'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = b"print('world')"
    var_9 = '/tmp/temp_script.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/script'
    var_1 = '.py'
    var_2 = 'Template error'
    var_3 = 'name'
    var_4 = 'world'
    var_5 = {var_3: var_4}
    var_6 = '/tmp/script.py'
    var_7 = '.'
    var_8 = module_0.run_script_with_context(var_6, var_7, var_5)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/script'
    var_1 = '.sh'
    var_2 = '/tmp/script.sh'
    var_3 = '.'
    var_4 = {}
    var_5 = module_0.run_script_with_context(var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir. Retrieved 4/22 statements.
# Partially parsed test_run_pre_prompt_hook_failure_raises_exception. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = f'{var_1}.py'
    var_3 = 'import sys; sys.exit(0)'
    var_4 = 'cookiecutter'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = f'{var_1}.py'
    var_3 = 'import sys; sys.exit(1)'
    var_4 = 'Pre-Prompt Hook script failed'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 2/7 statements.


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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_valid_hook_success. Retrieved 9/13 statements.
# Partially parsed test_valid_hook_name_mismatch. Retrieved 8/11 statements.
# Partially parsed test_valid_hook_unsupported_type. Retrieved 8/11 statements.
# Partially parsed test_valid_hook_is_backup_file. Retrieved 7/10 statements.
# Partially parsed test_valid_hook_complete_failure. Retrieved 8/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-checkout'
    var_2 = [var_0, var_1]
    var_3 = 'path/to/pre-commit.py'
    var_4 = 'pre-commit.py'
    var_5 = '.py'
    var_6 = ''
    var_7 = [var_0]
    var_8 = module_0.valid_hook(var_3, var_0)
    assert var_8 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = 'wrong-name.py'
    var_3 = 'wrong-name'
    var_4 = '.py'
    var_5 = (var_3, var_4)
    var_6 = 'path/to/wrong-name.py'
    var_7 = module_0.valid_hook(var_6, var_0)
    assert var_7 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = 'unknown.py'
    var_3 = 'unknown'
    var_4 = '.py'
    var_5 = (var_3, var_4)
    var_6 = 'path/to/unknown.py'
    var_7 = module_0.valid_hook(var_6, var_3)
    assert var_7 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = 'pre-commit.py~'
    var_3 = ''
    var_4 = (var_2, var_3)
    var_5 = 'path/to/pre-commit.py~'
    var_6 = module_0.valid_hook(var_5, var_0)
    assert var_6 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = []
    var_1 = 'invalid.py'
    var_2 = 'invalid'
    var_3 = '.py'
    var_4 = (var_2, var_3)
    var_5 = 'path/to/invalid.py'
    var_6 = 'pre-commit'
    var_7 = module_0.valid_hook(var_5, var_6)
    assert var_7 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/10 statements.
# Partially parsed test_run_hook_executes_scripts. Retrieved 10/15 statements.
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
    var_0 = '/tmp/project/hooks/pre_gen_project_script'
    var_1 = [var_0]
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/tmp/project'
    var_7 = module_0.run_hook(var_5, var_6, var_4)
    var_8 = 0
    var_9 = var_1[var_8]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/project/hooks/pre_gen_project_1'
    var_1 = '/tmp/project/hooks/pre_gen_project_2'
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

# Partially parsed test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec. Retrieved 4/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed (error:'
    var_5 = bool('Hook script failed (error:' in var_3)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_valid_hook_found. Retrieved 8/20 statements.
# Partially parsed test_find_hook_ignores_invalid_hooks. Retrieved 10/24 statements.
# Partially parsed test_find_hook_multiple_valid_hooks. Retrieved 9/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre-commit'

def test_case_0():
    var_0 = 'your_module._HOOKS'
    var_1 = 'pre-commit'
    assert var_1 == 1
    var_2 = 'post-merge'
    var_3 = [var_1, var_2]
    var_4 = 'pre-commit.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = 'pre-commit'
    var_7 = 0

def test_case_0():
    var_0 = 'your_module._HOOKS'
    var_1 = 'pre-commit'
    var_2 = [var_1]
    var_3 = 'unknown-hook.sh'
    var_4 = 'pre-commit.sh~'
    var_5 = 'post-merge.sh'
    var_6 = ''
    var_7 = ''
    var_8 = ''
    var_9 = 'pre-commit'

def test_case_0():
    var_0 = 'your_module._HOOKS'
    var_1 = 'pre-commit'
    assert var_1 == 2
    var_2 = 'post-merge'
    var_3 = [var_1, var_2]
    var_4 = 'pre-commit.sh'
    var_5 = 'pre-commit.py'
    var_6 = ''
    var_7 = ''
    var_8 = 'pre-commit'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_find_hook_signature_type_validity.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_hook_returns_absolute_paths_for_valid_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_multiple_paths_when_multiple_valid_hooks_exist. Retrieved 6/15 statements.
# Partially parsed test_find_hook_filters_out_invalid_hooks. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre-commit.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'some_name'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'pre-commit.py'
    var_5 = 'post-checkout.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'any_name'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'valid.py'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/12 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_windows_shell_true. Retrieved 4/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/usr/bin/script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = True
    var_3 = '.'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/17 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_deletes_project. Retrieved 8/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_does_not_delete_project. Retrieved 7/18 statements.


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
    var_1 = [var_0]
    var_2 = {}
    var_3 = '/repo'
    var_4 = 'post_gen_project'
    var_5 = '/project'
    var_6 = {}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)
    var_9 = '/project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Failed'
    var_1 = [var_0]
    var_2 = {}
    var_3 = '/repo'
    var_4 = 'post_gen_project'
    var_5 = '/project'
    var_6 = {}
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'empty_repo'



# Parsed testcases at query #15
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
    var_2 = '/path/to/post-checkout.py'
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
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = '/path/to/pre-commit.txt'
    var_3 = module_0.valid_hook(var_2, var_0)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = [var_0]
    var_2 = ''
    var_3 = module_0.valid_hook(var_2, var_2)
    assert var_3 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_signature_validity. Retrieved 7/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = None
    var_4 = lambda : var_3
    var_5 = type(var_4)
    var_6 = '__call__'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec. Retrieved 4/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed (error:'
    var_5 = bool('Hook script failed (error:' in var_3)
    assert var_5 is True



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

# Partially parsed test_find_hook_signature_is_correct. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'hook_name'
    var_1 = 'hook_name'
    var_2 = None
    var_3 = type(var_2)
    var_4 = 'hooks_dir'



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/16 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 5/14 statements.
# Partially parsed test_run_script_windows_shell_true. Retrieved 4/14 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/12 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 4/12 statements.
# Partially parsed test_run_script_os_error_generic. Retrieved 5/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test.py'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'might be an empty file or missing a shebang'
    var_5 = bool('might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'No such file'
    var_2 = OSError(var_0, var_1)
    var_3 = 'test.py'
    var_4 = module_0.run_script(var_3)
    var_5 = 'Hook script failed (error: [Errno 2] No such file)'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_script_with_context_skips_temp_file_creation_when_extension_is_empty. Retrieved 7/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'myscript'
    var_1 = '/tmp/cwd'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'hello world'
    var_6 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/16 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/14 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/11 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 4/11 statements.
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
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/tmp/test.py'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = OSError(var_0)
    var_2 = '/tmp/test.py'
    var_3 = module_0.run_script(var_2)
    var_4 = str(var_3)
    var_5 = 'Hook script failed (error: Permission denied)'
    var_6 = bool('Hook script failed (error: Permission denied)' in var_4)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_hook_returns_none_when_directory_is_empty. Retrieved 1/5 statements.
# Partially parsed test_find_hook_returns_path_for_valid_matching_hook. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/10 statements.
# Partially parsed test_find_hook_ignores_mismatched_hook_names. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_directory_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit.sh'
    var_1 = '#!/bin/bash\n'
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit.sh~'
    var_1 = ''
    var_2 = 'pre-commit'

def test_case_0():
    var_0 = 'post-commit.sh'
    var_1 = ''
    var_2 = 'pre-commit'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_enoexec. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_success. Retrieved 5/20 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/15 statements.
# Partially parsed test_find_hook_ignores_mismatched_name. Retrieved 5/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh'
    assert var_1 == 1
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'pre-commit'
    var_4 = 0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh~'
    var_2 = 'backup content'
    var_3 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-merge'
    var_2 = 'post-merge.sh'
    var_3 = 'content'
    var_4 = 'pre-commit'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/12 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/11 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 2/9 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/9 statements.
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
    var_1 = module_0.run_script(var_0)
    var_2 = [var_0]
    var_3 = True
    var_4 = '.'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = 'might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed (error: [Errno 13] Permission denied)'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 7/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_hook_returns_none_when_directory_is_empty. Retrieved 1/6 statements.
# Partially parsed test_find_hook_returns_absolute_path_for_valid_hook. Retrieved 4/14 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/9 statements.
# Partially parsed test_find_hook_ignores_mismatched_hook_names. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir_12345'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre-commit'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit.sh~'
    var_2 = 'backup content'

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'post-commit.sh'
    var_2 = 'content'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_returns_original_dir.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir. Retrieved 4/26 statements.
# Partially parsed test_run_pre_prompt_hook_fails_on_bad_script. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'hooks'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'hooks'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = 493
    var_4 = 'FailedHookException was not raised'
    var_5 = AssertionError(var_4)
    var_6 = bool(var_0)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'
    var_8 = bool('Pre-Prompt Hook script failed' in var_4)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_find_hook_signature_is_correct.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_script_py_file_success. Retrieved 4/16 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/14 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 3/11 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 4/11 statements.
# Partially parsed test_run_script_os_error_generic. Retrieved 5/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.run_script(var_0)
    var_2 = str(var_0)
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)
    assert var_4 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = OSError(var_0)
    var_2 = 'test.py'
    var_3 = module_0.run_script(var_2)
    var_4 = str(var_3)
    var_5 = 'Hook script failed (error: Permission denied)'
    var_6 = bool('Hook script failed (error: Permission denied)' in var_4)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec. Retrieved 4/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed (error:'
    var_5 = bool('Hook script failed (error:' in var_3)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_script_raises_enoexec_error. Retrieved 4/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_hook_signature_type_hints. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_script_raises_failed_hook_exception_on_enoexec. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = str(var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True



