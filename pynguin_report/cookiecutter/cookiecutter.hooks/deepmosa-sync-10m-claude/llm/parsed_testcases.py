####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/post-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unsupported-hook'
    var_1 = 'unsupported-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/file'
    var_1 = 'non-existent'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/very/long/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/Pre-Commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 16/32 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter.hooks.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter'
    var_9 = 'name'
    var_10 = 'test_project'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = len(var_3)
    assert var_13 == 1
    var_14 = var_3[0][1]
    var_15 = 0
    var_16 = var_3[var_15][var_15]
    var_17 = [var_16]
    var_18 = 'test_project'
    var_19 = '{{ cookiecutter.name }}'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 11/22 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 11/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 11/23 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/18 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/15 statements.
# Partially parsed test_run_script_make_executable_called. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')\n"
    var_2 = 'obj'
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda : var_4
    var_6 = {var_3: var_5}
    var_7 = 'subprocess.Popen'
    var_8 = 'utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'\n"
    var_2 = 'obj'
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda : var_4
    var_6 = {var_3: var_5}
    var_7 = 'subprocess.Popen'
    var_8 = 'utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys\nsys.exit(1)\n'
    var_2 = 'obj'
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda : var_4
    var_6 = {var_3: var_5}
    var_7 = 'subprocess.Popen'
    var_8 = 'utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script'
    var_1 = ''
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subprocess.Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Permission denied'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')\n"
    var_2 = []
    var_3 = 'obj'
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda : var_5
    var_7 = {var_4: var_6}
    var_8 = 'subprocess.Popen'
    var_9 = 'utils.make_executable'
    var_10 = len(var_2)
    assert var_10 == 1
    var_11 = var_2[0]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = 'pre-commit'
    var_3 = 0
    var_4 = {var_0}
    var_5 = '~'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_windows_uses_shell. Retrieved 6/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/18 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/18 statements.
# Partially parsed test_run_script_with_path_object. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][1]['shell']
    assert var_7 is False

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = var_2[0][1]['shell']
    assert var_6 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error'
    var_7 = bool('error' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_no_delete. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = None
    var_5 = 'cookiecutter.hooks.run_hook'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'post_gen_project'
    var_10 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_script_path_endswith_py. Retrieved 3/15 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = '.py'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/16 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 15/22 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 19/26 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 11/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when a single hook script is found and executed.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = '/hooks/pre_prompt.sh'
    var_3 = [var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'
    var_12 = '/project'
    var_13 = module_0.run_hook(var_11, var_12, var_10)
    var_14 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when multiple hook scripts are found.'
    var_1 = '/hooks/post_gen_project.sh'
    var_2 = '/hooks/post_gen_project.py'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = '/project'
    var_14 = module_0.run_hook(var_12, var_13, var_11)
    var_15 = 0
    var_16 = var_3[var_15]
    var_17 = 1
    var_18 = var_3[var_17]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when hook scripts list is empty.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No %s hook found'



# Parsed testcases at query #9
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/home/user/.git/hooks/commit-msg'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid-hook'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid-hook~'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.sh~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/.bashrc'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/19 statements.
# Partially parsed test_run_hook_single_script_found. Retrieved 12/22 statements.
# Partially parsed test_run_hook_multiple_scripts_found. Retrieved 13/26 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_prompt'
    var_11 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook when a single hook script is found.'
    var_1 = 'pre_prompt.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_prompt'
    var_11 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook when multiple hook scripts are found.'
    var_1 = 'pre_prompt_1.sh'
    var_2 = 'pre_prompt_2.sh'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'post_gen_project'
    var_12 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns an empty list.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'
    var_11 = 'No %s hook found'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_found. Retrieved 3/8 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 8/18 statements.
# Partially parsed test_run_pre_prompt_hook_hook_failure. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_hooks. Retrieved 10/22 statements.
# Partially parsed test_run_pre_prompt_hook_with_string_path. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no hook is found.'
    var_1 = 'template'
    var_2 = 'hooks'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid pre_prompt hook.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# valid hook'
    var_5 = []
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when hook fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# hook that fails'
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt hooks.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# hook 1'
    var_5 = 'pre_prompt.sh'
    var_6 = '#!/bin/bash\necho test'
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 2

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook works with string path.'
    var_1 = 'template'
    var_2 = 'hooks'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 22/29 statements.
# Partially parsed test_find_hook_with_nonexistent_hooks_dir. Retrieved 6/7 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 8/14 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 20/29 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 20/26 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 25/32 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'os.listdir'
    var_4 = 'pre_prompt.py'
    var_5 = [var_4]
    var_6 = lambda x: var_5
    var_7 = 'os.path.basename'
    var_8 = lambda x: var_4
    var_9 = 'os.path.splitext'
    var_10 = 'pre_prompt'
    var_11 = '.py'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = 'os.path.abspath'
    var_15 = -1
    var_16 = '/'
    var_17 = 'os.path.join'
    var_18 = lambda x, y: f'{x}/{y}'
    var_19 = 'hooks'
    var_20 = module_0.find_hook(var_10, var_19)
    var_21 = bool(var_20 is not None)
    assert var_21 is True
    var_22 = len(var_20)
    var_23 = bool(var_22 > 0)
    assert var_23 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = 'pre_prompt'
    var_4 = 'nonexistent_hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'os.path.isdir'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'os.listdir'
    var_5 = []
    var_6 = lambda x: var_5
    var_7 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'os.listdir'
    var_4 = 'pre_prompt.py~'
    var_5 = [var_4]
    var_6 = lambda x: var_5
    var_7 = 'os.path.basename'
    var_8 = lambda x: var_4
    var_9 = 'os.path.splitext'
    var_10 = 'pre_prompt.py'
    var_11 = '~'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = 'os.path.abspath'
    var_15 = 'os.path.join'
    var_16 = lambda x, y: f'{x}/{y}'
    var_17 = 'pre_prompt'
    var_18 = 'hooks'
    var_19 = module_0.find_hook(var_17, var_18)
    assert var_19 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'os.listdir'
    var_4 = 'unsupported_hook.py'
    var_5 = [var_4]
    var_6 = lambda x: var_5
    var_7 = 'os.path.basename'
    var_8 = lambda x: var_4
    var_9 = 'os.path.splitext'
    var_10 = 'unsupported_hook'
    var_11 = '.py'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = 'os.path.abspath'
    var_15 = lambda x: x
    var_16 = 'os.path.join'
    var_17 = lambda x, y: f'{x}/{y}'
    var_18 = 'hooks'
    var_19 = module_0.find_hook(var_10, var_18)
    assert var_19 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'os.listdir'
    var_4 = 'pre_prompt.py'
    var_5 = 'pre_prompt.sh'
    var_6 = [var_4, var_5]
    var_7 = lambda x: var_6
    var_8 = 'os.path.basename'
    var_9 = -1
    var_10 = '/'
    var_11 = lambda x: x.split(var_10)[var_9]
    var_12 = 'os.path.splitext'
    var_13 = 0
    var_14 = '.'
    var_15 = -1
    var_16 = lambda x: (x.rsplit(var_14, var_1)[var_13], var_14 + x.rsplit(var_14, var_1)[var_15])
    var_17 = 'os.path.abspath'
    var_18 = -1
    var_19 = 'os.path.join'
    var_20 = lambda x, y: f'{x}/{y}'
    var_21 = 'pre_prompt'
    var_22 = 'hooks'
    var_23 = module_0.find_hook(var_21, var_22)
    var_24 = bool(var_23 is not None)
    assert var_24 is True
    var_25 = len(var_23)
    assert var_25 == 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 9/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile.NamedTemporaryFile is called with delete=False.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/path/to/script.sh'
    var_6 = '/cwd'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 8/24 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 8/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/17 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/18 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/15 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1
    var_8 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1
    var_8 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'custom'
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = var_2[0][1]['cwd']



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda x: var_2
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = '.'
    var_9 = module_0.run_hook(var_7, var_8, var_6)
    var_10 = 'No pre_prompt hook found'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 16/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to False when scripts exist.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = "#!/usr/bin/env python\nprint('test')"
    var_4 = 'os.path.isdir'
    var_5 = True
    var_6 = 'os.listdir'
    var_7 = [var_2]
    var_8 = 'os.path.abspath'
    var_9 = lambda x: str(x)
    var_10 = 'os.path.join'
    var_11 = lambda a, b: f'{a}/{b}'
    var_12 = 'valid_hook'
    var_13 = 'pre_prompt'
    var_14 = module_0.find_hook(var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True
    var_16 = len(var_14)
    var_17 = bool(var_16 > 0)
    assert var_17 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_returns_scripts_list_when_valid_hooks_found. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_script. Retrieved 9/21 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_exception. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_hook_scripts. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt hook script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when hook script raises FailedHookException.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with multiple pre_prompt hook scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test1'"
    var_5 = 493
    var_6 = 'pre_prompt.py'
    var_7 = "print('test2')"
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_hook_predicate_false. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = f'{var_0}.py'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.work_in'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 11/29 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'test_hook'
    var_4 = 'test_hook.sh'
    var_5 = '#!/bin/bash\n'
    var_6 = 'test_hook'
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = len(var_2)
    var_9 = bool(var_8 > 0)
    assert var_9 is True
    var_10 = 'nonexistent_hook'
    var_11 = all(var_8)
    var_12 = var_1 and var_11
    var_13 = bool(var_2 is None or var_12)
    assert var_13 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'pre_prompt'
    var_5 = '.'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No pre_prompt hook found'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_no_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without errors.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter.hooks.rmtree'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = True
    var_13 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'pre_gen_project'
    var_12 = False
    var_13 = 'pre_gen_project'

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_prompt'
    var_12 = True
    var_13 = 'post_prompt'

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on UndefinedError when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'pre_gen_project'
    var_12 = False
    var_13 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = True
    var_10 = 0
    var_11 = var_3[var_10]
    var_12 = str(var_11)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_nonexistent_hooks_dir. Retrieved 3/4 statements.
# Partially parsed test_find_hook_with_no_matching_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 6/18 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py~'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project.sh'
    var_4 = '#!/bin/bash\n'
    var_5 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 8/17 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 8/18 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 8/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'test_hook'
    var_5 = 'hooks'
    var_6 = 'test_hook'
    var_7 = all(var_5)
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = None
    var_8 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_9 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 14/22 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 14/20 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 14/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/15 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/15 statements.
# Partially parsed test_run_script_calls_make_executable. Retrieved 13/21 statements.
# Partially parsed test_run_script_python_file_uses_sys_executable. Retrieved 6/16 statements.
# Partially parsed test_run_script_shell_script_direct_command. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Permission denied'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = lambda *args, **kwargs: var_8
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = var_1[0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = var_1[0]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with the correct suffix from script_path.'
    var_1 = '/path/to/script.sh'
    var_2 = '/tmp'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.run_script_with_context(var_1, var_2, var_7)
    var_9 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_oserror_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 13/29 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = len(var_6)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_6[var_10]
    var_12 = str(var_11)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_find_hook_predicate_evaluates_to_false. Retrieved 5/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_file.txt'
    var_2 = 'content'
    var_3 = 'test_hook'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Variable undefined'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook from repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'os.getcwd'
    var_9 = 'os.chdir'
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 12/22 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 12/20 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 12/21 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 3/15 statements.
# Partially parsed test_run_script_os_error_other. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash'
    var_2 = 'Popen'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Permission denied'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []
    var_4 = lambda hook_name: var_3



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'Exec format error'
    var_1 = '/path/to/script.sh'
    var_2 = [var_1]
    var_3 = False
    var_4 = '.'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_directory. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on failure when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = None
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'pre_prompt'
    var_11 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 13/35 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the delete parameter in NamedTemporaryFile is False.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_jinja2_env_vars'
    var_4 = 'test_project'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = "#!/bin/bash\necho 'test'"
    var_9 = '/tmp/script.sh'
    var_10 = '/tmp'
    var_11 = module_0.run_script_with_context(var_9, var_10, var_7)
    var_12 = 1



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used at line 7 of run_pre_prompt_hook.'
    var_1 = None
    var_2 = False
    assert var_2 is True
    var_3 = None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 10/27 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 10/27 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'pre_prompt'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'pre_prompt'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'pre_prompt'
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(not project_dir.exists())
    assert var_11 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_found. Retrieved 5/9 statements.
# Partially parsed test_run_pre_prompt_hook_executes_script. Retrieved 11/27 statements.
# Partially parsed test_run_pre_prompt_hook_failed_exception. Retrieved 6/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_repo. Retrieved 6/21 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no hook is found.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = None
    var_4 = lambda *args, **kwargs: var_3

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temp dir and executes script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook executed')"
    var_5 = []
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_9 = lambda x: x
    var_10 = len(var_5)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException on script failure.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_5 = lambda x: x
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary repository.'
    var_1 = 'test_repo'
    var_2 = 'temp_repo'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter.hooks.create_tmp_repo_dir'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple hook scripts.'
    var_1 = 'test_repo'
    var_2 = []
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_6 = lambda x: x
    var_7 = len(var_2)
    assert var_7 == 2
    var_8 = bool(var_2 == ['/script1.py', '/script2.sh'])
    assert var_8 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'test_script.py'
    var_5 = False
    var_6 = '.'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_pre_prompt_scripts. Retrieved 5/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = '/test/repo'
    var_2 = None
    var_3 = module_0.run_pre_prompt_hook(var_1)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True
    var_5 = 'pre_prompt'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_extension. Retrieved 9/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with delete=False and correct suffix.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/path/to/script.sh'
    var_6 = '/cwd'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_found. Retrieved 5/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 11/28 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 8/23 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_hooks. Retrieved 8/18 statements.
# Partially parsed test_run_pre_prompt_hook_creates_tmp_dir. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no hook found.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = None
    var_4 = lambda *args, **kwargs: var_3

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes hook script successfully.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_10 = lambda x: x

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = lambda x: x
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple hook scripts.'
    var_1 = 'test_repo'
    var_2 = []
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_6 = lambda x: x
    var_7 = len(var_2)
    assert var_7 == 2
    var_8 = '/path/to/script1.py'
    var_9 = bool('/path/to/script1.py' in var_2)
    assert var_9 is True
    var_10 = '/path/to/script2.py'
    var_11 = bool('/path/to/script2.py' in var_2)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temporary directory when hook exists.'
    var_1 = 'test_repo'
    var_2 = 'tmp_repo'
    var_3 = 0
    var_4 = [var_3]
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = 'cookiecutter.hooks.create_tmp_repo_dir'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 9/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 14 (delete=False) evaluates to False.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/script.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_script_python_file. Retrieved 4/16 statements.
# Partially parsed test_run_script_non_python_file. Retrieved 3/14 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 3/13 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 4/14 statements.
# Partially parsed test_run_script_os_error_other. Retrieved 5/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/tmp'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '.'
    var_2 = 'Exec format error'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = 2
    var_3 = 'No such file or directory'
    var_4 = module_0.run_script(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-push'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unsupported-hook'
    var_1 = 'unsupported-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/home/user/.git/hooks/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 15/33 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 13/28 statements.
# Partially parsed test_run_script_with_context_with_jinja_variables. Retrieved 17/31 statements.
# Partially parsed test_run_script_with_context_passes_cwd. Retrieved 13/29 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.project_name }}')"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'temp_script.py'
    var_5 = None
    var_6 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_jinja2_env_vars'
    var_10 = 'my_project'
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = 0
    var_15 = b"print('my_project')"

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.message }}'"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = {}
    var_5 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_6 = 'cookiecutter'
    var_7 = 'message'
    var_8 = '_jinja2_env_vars'
    var_9 = 'Hello World'
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = var_4['suffix']
    assert var_13 == '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context correctly renders Jinja variables.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.name }} - {{ cookiecutter.version }}')"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'temp_script.py'
    var_5 = None
    var_6 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'version'
    var_10 = '_jinja2_env_vars'
    var_11 = 'TestProject'
    var_12 = '1.0.0'
    var_13 = {}
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = {var_7: var_14}
    var_16 = 0
    var_17 = b"print('TestProject - 1.0.0')"

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct cwd to run_script.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'temp_script.py'
    var_5 = None
    var_6 = 'cookiecutter.hooks.tempfile.NamedTemporaryFile'
    var_7 = 'cookiecutter'
    var_8 = '_jinja2_env_vars'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'work_dir'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_with_python_script. Retrieved 8/18 statements.
# Partially parsed test_run_pre_prompt_hook_returns_path_object. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt scripts exist.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda script, cwd: var_7

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a Python pre_prompt script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = None
    var_7 = lambda script, cwd: var_6

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a Path object when scripts exist.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda script, cwd: var_7



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/14 statements.
# Partially parsed test_run_hook_scripts_found_and_executed. Retrieved 16/22 statements.
# Partially parsed test_run_hook_single_script. Retrieved 14/18 statements.
# Partially parsed test_run_hook_with_pathlib_path. Retrieved 11/18 statements.
# Partially parsed test_run_hook_multiple_hooks_same_name. Retrieved 16/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when hook scripts are found and executed.'
    var_1 = '/hooks/pre_prompt.py'
    var_2 = '/hooks/pre_prompt.sh'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_prompt'
    var_13 = '/project'
    var_14 = module_0.run_hook(var_12, var_13, var_11)
    var_15 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook with a single hook script.'
    var_1 = '/hooks/post_gen_project.py'
    var_2 = [var_1]
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'post_gen_project'
    var_12 = '/my/project'
    var_13 = module_0.run_hook(var_11, var_12, var_10)

def test_case_0():
    var_0 = 'Test run_hook accepts Path objects for project_dir.'
    var_1 = '/hooks/pre_prompt.sh'
    var_2 = [var_1]
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = '/project/path'
    var_10 = [var_9]
    var_11 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes all scripts when multiple hooks have the same name.'
    var_1 = '/hooks/pre_prompt.py'
    var_2 = '/hooks/pre_prompt.sh'
    var_3 = '/hooks/pre_prompt.bash'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = 'cookiecutter.hooks.run_script_with_context'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'test'
    var_10 = 'data'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_prompt'
    var_14 = '/project'
    var_15 = module_0.run_hook(var_13, var_14, var_12)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/14 statements.
# Partially parsed test_run_hook_scripts_found_and_executed. Retrieved 15/21 statements.
# Partially parsed test_run_hook_multiple_scripts_executed. Retrieved 15/20 statements.
# Partially parsed test_run_hook_with_different_hook_names. Retrieved 12/17 statements.
# Partially parsed test_run_hook_with_path_object. Retrieved 11/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No %s hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when hook scripts are found and executed.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = '/hooks/pre_prompt.sh'
    var_3 = [var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'
    var_12 = '/project'
    var_13 = module_0.run_hook(var_11, var_12, var_10)
    var_14 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts in order.'
    var_1 = '/hooks/pre_prompt.sh'
    var_2 = '/hooks/pre_prompt.py'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_prompt'
    var_13 = '/project'
    var_14 = module_0.run_hook(var_12, var_13, var_11)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook with different hook names.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = '/hooks/post_gen_project.sh'
    var_3 = [var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'post_gen_project'
    var_10 = '/project'
    var_11 = module_0.run_hook(var_9, var_10, var_8)

def test_case_0():
    var_0 = 'Test run_hook accepts Path object as project_dir.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = '/hooks/pre_prompt.sh'
    var_3 = [var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = '/project'
    var_10 = [var_9]
    var_11 = 'pre_prompt'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt scripts exist.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = 'cookiecutter'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 4/9 statements.
# Partially parsed test_run_pre_prompt_hook_executes_script. Retrieved 6/20 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script_raises_exception. Retrieved 6/15 statements.
# Partially parsed test_run_pre_prompt_hook_python_script. Retrieved 5/18 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.json'
    var_3 = '{}'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temp dir and runs pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test' > output.txt"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes Python pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test1'"
    var_5 = 493
    var_6 = 'pre_prompt.py'
    var_7 = "print('test2')"



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true. Retrieved 5/31 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'subprocess.Popen'
    var_2 = 'test_script.py'
    var_3 = 'win'
    var_4 = '.'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/4 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_none_when_hooks_dir_is_empty. Retrieved 2/7 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 6/16 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/14 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_different_extensions. Retrieved 8/21 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 6/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = {var_4}

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = '#!/usr/bin/env python'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = {var_4}

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.sh'
    var_3 = '#!/usr/bin/env python'
    var_4 = '#!/bin/bash'
    var_5 = '__main__._HOOKS'
    var_6 = 'pre_prompt'
    var_7 = {var_6}

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = {var_4}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'test_repo'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit'
    var_2 = 'post-commit'
    var_3 = 'commit-msg'
    var_4 = {var_1, var_2, var_3}
    var_5 = True
    var_6 = True
    var_7 = False
    var_8 = var_5 and var_6 and var_1
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_list_with_single_matching_hook. Retrieved 7/16 statements.
# Partially parsed test_find_hook_returns_list_with_multiple_matching_hooks. Retrieved 9/21 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 8/20 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 8/15 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash'
    var_4 = '#!/usr/bin/env python'
    var_5 = '__main__._HOOKS'
    var_6 = 'pre_prompt'
    var_7 = 'post_gen_project'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.sh~'
    var_3 = '#!/bin/bash'
    var_4 = '__main__._HOOKS'
    var_5 = 'pre_prompt'
    var_6 = 'post_gen_project'
    var_7 = [var_5, var_6]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.sh'
    var_2 = '#!/bin/bash'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]
    var_7 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash'
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 6/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = 'pre_prompt'
    var_3 = '.'
    var_4 = {}
    var_5 = module_0.run_hook(var_2, var_3, var_4)
    assert var_5 is None
    var_6 = 'No pre_prompt hook found'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '#!/bin/bash\n'
    var_2 = 'pre-commit'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 17/33 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    assert var_1 is None
    var_2 = None
    var_3 = []
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 == var_5
    var_7 = None
    var_8 = var_7 if var_6 else var_3
    assert var_8 is None
    var_9 = None
    var_10 = '/abs/hooks/hook_script.sh'
    var_11 = [var_10]
    var_12 = len(var_11)
    var_13 = 0
    var_14 = var_12 > var_13
    var_15 = None
    var_16 = var_11 if var_14 else var_15
    var_17 = bool(var_10 or var_16 is None)
    assert var_17 is True



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = module_0.valid_hook(var_1, var_0)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_with_cleanup. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_cleanup. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without cleanup.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.rmtree'
    var_8 = 'post_gen_project'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'post_gen_project'
    var_12 = True

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't cleanup when delete_project_on_failure is False."
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'post_gen_project'
    var_11 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Undefined variable'
    var_8 = module_0.UndefinedError(var_7)
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = str(var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_pre_prompt_hook. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_pre_prompt_hook. Retrieved 5/15 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_directory. Retrieved 5/15 statements.
# Partially parsed test_run_pre_prompt_hook_preserves_repo_structure. Retrieved 7/21 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_raises_exception. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_with_string_path. Retrieved 2/7 statements.
# Partially parsed test_run_pre_prompt_hook_with_path_object. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt hook and returns new repo_dir.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when hook exists.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook preserves repo directory structure.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "test"}'
    var_5 = 'pre_prompt.py'
    var_6 = "print('hook')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when hook fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook accepts string path as repo_dir.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook accepts Path object as repo_dir.'
    var_1 = 'template'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = []
    var_2 = 'cookiecutter.hooks.logger'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = '/tmp/test_project'
    var_8 = [var_7]
    var_9 = 'pre_prompt'
    var_10 = 'No %s hook found'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that find_hook returns None when hooks directory does not exist.'
    var_1 = 'non_existent_hooks'
    var_2 = 'post_gen_project'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = 'post_gen_project'
    var_8 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/25 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_renders_jinja_variables. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_passes_cwd. Retrieved 11/22 statements.
# Partially parsed test_run_script_with_context_handles_empty_context. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 0
    assert var_11 == 1
    var_12 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_name'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = None
    assert var_11 == '.py'
    var_12 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context properly renders Jinja2 variables.'
    var_1 = "#!/bin/bash\necho 'Project: {{ cookiecutter.project_name }}'\necho 'Author: {{ cookiecutter.author }}'"
    var_2 = 'render_test.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = '_jinja2_env_vars'
    var_8 = 'awesome_project'
    var_9 = 'John Doe'
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = None
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = 'awesome_project'
    var_16 = bool('awesome_project' in var_13)
    assert var_16 is True
    var_17 = 'John Doe'
    var_18 = bool('John Doe' in var_13)
    assert var_18 is True

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct working directory.'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = 'cwd_test.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = '_jinja2_env_vars'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = None
    var_10 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context handles minimal context.'
    var_1 = "#!/bin/bash\necho 'static content'"
    var_2 = 'empty_context.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = '_jinja2_env_vars'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = False
    assert var_9 is True
    var_10 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 14/31 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = None
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'post_gen_project.py'
    var_12 = False
    var_13 = len(var_6)
    var_14 = bool(var_13 > 0)
    assert var_14 is True
    var_15 = var_6[0]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_predicate_evaluates_to_false. Retrieved 6/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__.valid_hook'
    var_4 = 'post_gen_project'
    var_5 = module_0.find_hook(var_4, var_0)
    assert var_5 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 14/27 statements.
# Partially parsed test_run_script_with_context_python_extension. Retrieved 14/24 statements.
# Partially parsed test_run_script_with_context_multiple_variables. Retrieved 15/25 statements.
# Partially parsed test_run_script_with_context_custom_cwd. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = var_11[0][1]

def test_case_0():
    var_0 = 'Test run_script_with_context with Python script.'
    var_1 = 'print("{{ cookiecutter.name }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_name'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = '{{ cookiecutter.var1 }} and {{ cookiecutter.var2 }}'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'var1'
    var_6 = 'var2'
    var_7 = '_jinja2_env_vars'
    var_8 = 'value1'
    var_9 = 'value2'
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = []
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = 'value1 and value2'
    var_16 = bool('value1 and value2' in var_13[0])
    assert var_16 is True

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct cwd to run_script.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'utf-8'
    var_4 = 'custom_dir'
    var_5 = 'cookiecutter'
    var_6 = '_jinja2_env_vars'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = var_10[0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/12 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 4/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/12 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 6/21 statements.
# Partially parsed test_find_hook_uses_default_hooks_dir. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = '#!/bin/bash'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash'
    var_4 = '#!/usr/bin/env python'
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 4/12 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/6 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_non_matching_hook_name. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_multiple_matching_files. Retrieved 6/19 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py~'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'post_gen_project.sh'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = '#!/bin/bash\n'
    var_5 = 'post_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 10/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that NamedTemporaryFile is created with delete=False at line 14.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = '/path/to/script.sh'
    var_7 = '/cwd'
    var_8 = module_0.run_script_with_context(var_6, var_7, var_4)
    var_9 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_oserror_errno_not_enoexec. Retrieved 3/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(str(e).startswith('Hook script failed (error:'))
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 5/22 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/19 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 7/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 7/21 statements.
# Partially parsed test_run_script_oserror. Retrieved 7/18 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 6/22 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = False
    var_6 = True
    var_7 = 'exit status: 1'
    var_8 = bool(var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = False
    var_6 = True
    var_7 = 'shebang'
    var_8 = bool(var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = False
    var_6 = True
    var_7 = 'error'
    var_8 = bool('error' in str(e).lower())
    assert var_8 is True
    var_9 = bool(var_6)
    assert var_9 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = var_2[0]
    assert var_6 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = var_2[0]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager_enters_repo_dir. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager enters the repo directory at line 7.'
    var_1 = 'test_repo'
    var_2 = None
    var_3 = 'chdir'
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = []
    var_6 = lambda x: var_5



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = 'test_repo'
    var_2 = 'test_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = 'post_gen_project'
    var_8 = False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_pre_prompt_hook_predicate_line_7_evaluates_to_false. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (if not scripts) evaluates to False.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 13/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_with_deletion. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'
    var_11 = False
    var_12 = len(var_3)
    assert var_12 == 1
    var_13 = var_3[0][0]
    assert var_13 == 'post_gen_project'
    var_14 = var_3[0][2]
    var_15 = bool(var_3[0][2] == var_9)
    assert var_15 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException without deletion.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on hook failure when enabled.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'
    var_10 = True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError without deletion.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when enabled.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'
    var_10 = True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'
    var_11 = False
    var_12 = var_3[var_11]
    var_13 = str(var_12)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 13/25 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 13/25 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 11/22 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/17 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = []
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = len(var_9)
    assert var_12 == 1
    var_13 = var_9[0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = []
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = len(var_9)
    assert var_12 == 1
    var_13 = var_9[0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/custom/dir'
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'
    var_13 = var_10[0][2]
    var_14 = bool(var_10[0][2] == var_1)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'error'
    var_5 = bool('error' in str(e).lower())
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without errors.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'Hook failed'
    var_11 = [var_10]
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'Hook failed'
    var_11 = [var_10]
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'Undefined variable'
    var_11 = module_0.UndefinedError(var_10)
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes working directory to repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = None
    var_10 = None
    var_11 = 'cookiecutter.hooks.run_hook'
    var_12 = False
    var_13 = str(var_10)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test that non-FailedHookException and non-UndefinedError exceptions are not caught.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Some other error'
    var_8 = 'cookiecutter.hooks.work_in'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_find_hook_predicate_line_1_evaluates_to_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = None
    var_1 = type(var_0)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_find_hook_predicate_evaluates_to_false. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_file.txt'
    var_2 = 'content'
    var_3 = 'nonexistent_hook'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_find_hook_predicate_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = bool(not (var_0 is None or var_1 is None))
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 22/36 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 22/34 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 22/35 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 13/27 statements.
# Partially parsed test_run_script_other_oserror. Retrieved 13/24 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = 'builtins.__import__'
    var_12 = 'utils'
    var_13 = 'MockUtils'
    var_14 = ()
    var_15 = 'make_executable'
    var_16 = None
    var_17 = lambda x: var_16
    var_18 = {var_15: var_17}
    var_19 = type(var_13, var_14, var_18)
    var_20 = var_19()
    var_21 = lambda name, *args, **kwargs: __import__(name) if name != var_12 else var_20

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = 'builtins.__import__'
    var_12 = 'utils'
    var_13 = 'MockUtils'
    var_14 = ()
    var_15 = 'make_executable'
    var_16 = None
    var_17 = lambda x: var_16
    var_18 = {var_15: var_17}
    var_19 = type(var_13, var_14, var_18)
    var_20 = var_19()
    var_21 = lambda name, *args, **kwargs: __import__(name) if name != var_12 else var_20

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = 'builtins.__import__'
    var_12 = 'utils'
    var_13 = 'MockUtils'
    var_14 = ()
    var_15 = 'make_executable'
    var_16 = None
    var_17 = lambda x: var_16
    var_18 = {var_15: var_17}
    var_19 = type(var_13, var_14, var_18)
    var_20 = var_19()
    var_21 = lambda name, *args, **kwargs: __import__(name) if name != var_12 else var_20
    var_22 = bool(False)
    assert var_22 is True
    var_23 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'builtins.__import__'
    var_3 = 'utils'
    var_4 = 'MockUtils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda name, *args, **kwargs: __import__(name) if name != var_3 else var_11
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'builtins.__import__'
    var_3 = 'utils'
    var_4 = 'MockUtils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda name, *args, **kwargs: __import__(name) if name != var_3 else var_11
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Hook script failed (error:'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir for execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test that predicate at line 20 evaluates to False when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Test error'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 15/25 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 15/24 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 15/26 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 14/21 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/17 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/14 statements.
# Partially parsed test_run_script_makes_executable. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockPopen'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = {}
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = 'cwd'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockPopen'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = 'Popen'
    var_10 = lambda *args, **kwargs: var_8
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error'
    var_7 = bool('error' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 15/25 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 13/24 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir successfully executes a hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Variable undefined'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = None
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_predicate_false. Retrieved 10/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 17 (with work_in) evaluates to False when dirname is None.'
    var_1 = None
    var_2 = 'post_gen_project'
    var_3 = '/tmp/project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = None
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_script_with_context_temp_file_delete_false. Retrieved 9/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 14 (delete=False) evaluates to False.'
    var_1 = '/tmp/test_script.sh'
    var_2 = '/tmp'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = module_0.run_script_with_context(var_1, var_2, var_5)
    var_8 = 1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 6/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = 'test_script.py'
    var_4 = '.'
    var_5 = module_0.run_script(var_3, var_4)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = '#!/bin/bash\necho "test"'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = bool(var_3)
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_false. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that non-matching exceptions are not caught at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(project_dir.exists())
    assert var_10 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_exit_status_not_equal_to_exit_success. Retrieved 3/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed (exit status: 1)'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_without_cleanup. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter.hooks.rmtree'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = True
    var_13 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = False
    var_13 = 'post_gen_project'

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = True
    var_13 = 'post_gen_project'

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on UndefinedError when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = False
    var_13 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'os.getcwd'
    var_4 = 'os.chdir'
    var_5 = 'cookiecutter.hooks.run_hook'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'post_gen_project'
    var_10 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/22 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 7/21 statements.
# Partially parsed test_run_script_windows_uses_shell. Retrieved 7/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/19 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/20 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/17 statements.
# Partially parsed test_run_script_cwd_parameter. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][1]['shell']
    assert var_7 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho test'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = "print('test')"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'sys.platform'
    var_6 = 'linux'
    var_7 = len(var_3)
    assert var_7 == 1
    var_8 = var_3[0][1]['cwd']



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_oserror_errno_not_enoexec. Retrieved 3/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = 'might be an empty file or missing a shebang'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 9/29 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'test_hook'
    var_4 = 'test_hook.sh'
    var_5 = '#!/bin/bash\n'
    var_6 = 'test_hook'
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = len(var_2)
    var_9 = bool(var_8 > 0)
    assert var_9 is True
    var_10 = 'nonexistent'
    var_11 = bool(var_2 is None or var_8)
    assert var_11 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 9/29 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with delete=False at line 14.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/path/to/script.sh'
    var_6 = '/cwd'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 7/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Popen'
    var_1 = 'make_executable'
    var_2 = None
    var_3 = lambda x: var_2
    var_4 = 'test_script.py'
    var_5 = '.'
    var_6 = module_0.run_script(var_4, var_5)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/9 statements.
# Partially parsed test_work_in_returns_to_original_directory_on_exception. Retrieved 4/13 statements.
# Partially parsed test_work_in_with_none_dirname. Retrieved 1/5 statements.
# Partially parsed test_work_in_with_path_object. Retrieved 2/9 statements.
# Partially parsed test_work_in_with_string_path. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager changes to the specified directory.'
    var_1 = 'test_subdir'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True
    var_3 = 'Test exception'
    var_4 = ValueError(var_3)

def test_case_0():
    var_0 = 'Test that work_in with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test that work_in accepts Path objects.'
    var_1 = 'test_subdir'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test that work_in accepts string paths.'
    var_1 = 'test_subdir'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 7/22 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 7/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/21 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/22 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/19 statements.
# Partially parsed test_run_script_custom_cwd. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][1]['shell']
    assert var_7 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('fail')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'subdir'
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'sys.platform'
    var_6 = 'linux'
    var_7 = len(var_3)
    assert var_7 == 1
    var_8 = var_3[0][1]['cwd']



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'Test that the delete parameter in NamedTemporaryFile is False.'
    var_1 = '/tmp/test_script.sh'
    var_2 = [var_1]
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 1



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 10/27 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 10/27 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 12/32 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.work_in'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'cookiecutter.hooks.work_in'
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'post_gen_project'
    var_11 = True
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/29 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 16/27 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 15/27 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 14/22 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/18 statements.
# Partially parsed test_run_script_os_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    assert var_15 == 1
    var_16 = var_10[0][0][0]
    var_17 = var_10[0][1]['cwd']
    assert var_17 == '.'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    assert var_15 == 1
    var_16 = var_10[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = var_10[0][1]['cwd']

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockPopen'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = var_7()
    var_9 = 'Popen'
    var_10 = lambda *args, **kwargs: var_8
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Permission denied'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_no_cleanup_on_failure. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'pre_prompt'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Variable undefined'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_prompt'
    var_11 = True

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't cleanup when delete_project_on_failure is False."
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook in correct working directory.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'pre_prompt'
    var_9 = False



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 12/22 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_prompt'
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Variable undefined'
    var_8 = module_0.UndefinedError(var_7)
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_prompt'
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test that project directory is deleted when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Exec format error'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'might be an empty file or missing a shebang'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts_found. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 8/18 statements.
# Partially parsed test_run_pre_prompt_hook_with_failed_hook. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_returns_path_object. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt hook script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook executed')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = None
    var_7 = lambda script, cwd: var_6

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when hook script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook executed')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns a Path when hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = None
    var_7 = lambda script, cwd: var_6



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found.'
    var_1 = None
    var_2 = True



# Parsed testcases at query #71
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.
# Failed to parse test_work_in_with_none_stays_in_current_directory.
# Partially parsed test_work_in_restores_directory_on_exception. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test exception'
    var_1 = ValueError(var_0)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_oserror_predicate_enoexec. Retrieved 1/6 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = '/tmp/test_project'
    var_8 = [var_7]
    var_9 = 'pre_prompt'
    var_10 = 'No pre_prompt hook found'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 11/35 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_prompt'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.sh'
    var_7 = 'w'
    var_8 = 'pre_prompt'
    var_9 = bool(var_1)
    assert var_9 is True
    var_10 = len(var_2)
    var_11 = bool(var_10 > 0)
    assert var_11 is True
    var_12 = all(var_7)
    var_13 = bool(var_12)
    assert var_13 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 13/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_jinja2_env_vars'
    var_4 = 'test_project'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = '#!/bin/bash\necho "test"'
    var_9 = '/path/to/script.sh'
    var_10 = '/cwd'
    var_11 = module_0.run_script_with_context(var_9, var_10, var_7)
    var_12 = 1



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/6 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 3/7 statements.
# Partially parsed test_find_hook_no_matching_hooks. Retrieved 5/11 statements.
# Partially parsed test_find_hook_single_matching_hook. Retrieved 5/14 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 7/20 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/11 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/11 statements.
# Partially parsed test_find_hook_mixed_files. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'Test find_hook when hooks directory does not exist.'
    var_1 = 'pre_prompt'
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = 'Test find_hook when hooks directory is empty.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook when no hooks match the hook_name.'
    var_1 = 'hooks'
    var_2 = 'other_hook.sh'
    var_3 = '#!/bin/bash\n'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with a single matching hook file.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = '#!/bin/bash\n'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with multiple matching hook files.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/bin/bash\n'
    var_5 = '#!/usr/bin/env python\n'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores backup files ending with ~.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh~'
    var_3 = '#!/bin/bash\n'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores unsupported hook names.'
    var_1 = 'hooks'
    var_2 = 'unsupported_hook.sh'
    var_3 = '#!/bin/bash\n'
    var_4 = 'unsupported_hook'

def test_case_0():
    var_0 = 'Test find_hook with a mix of valid, invalid, and backup files.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt.sh~'
    var_4 = 'other_hook.sh'
    var_5 = '#!/bin/bash\n'
    var_6 = 'pre_prompt'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 8/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None
    var_2 = module_0.find_hook(var_0)
    assert var_2 is None
    var_3 = 'hook1.sh'
    var_4 = 'hook2.py'
    var_5 = True
    var_6 = module_0.find_hook(var_0)
    var_7 = len(var_6)
    assert var_7 == 2



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_find_hook_no_hooks_directory. Retrieved 3/4 statements.
# Partially parsed test_find_hook_empty_hooks_directory. Retrieved 2/7 statements.
# Partially parsed test_find_hook_matching_hook_found. Retrieved 4/13 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 6/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/11 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("test")'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.sh'
    var_3 = 'print("test")'
    var_4 = 'echo "test"'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'print("test")'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = 'print("test")'
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_prompt.py'
    var_2 = 'print("test")'
    var_3 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("test")'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 7/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    var_2 = 'some_file.sh'
    var_3 = module_0.find_hook(var_0)
    var_4 = 'test_hook.sh'
    var_5 = 'other.sh'
    var_6 = module_0.find_hook(var_0)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = f'{var_0}'
    var_2 = '#!/bin/bash\n'



