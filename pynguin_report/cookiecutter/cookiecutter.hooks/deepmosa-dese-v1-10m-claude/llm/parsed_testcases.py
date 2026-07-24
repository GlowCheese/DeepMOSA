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
    var_0 = '/path/to/invalid-hook'
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
    var_0 = '/different/path/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = 'pre-commit'
    var_3 = 'pre-commit'
    var_4 = var_3 == var_0
    var_5 = 'pre-commit'
    var_6 = 'post-commit'
    var_7 = [var_5, var_6]
    var_8 = var_3 in var_7
    var_9 = '~'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 6/17 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/12 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 7/20 statements.
# Partially parsed test_find_hook_uses_default_hooks_dir. Retrieved 7/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test find_hook returns None when hooks directory doesn't exist."
    var_1 = 'pre_prompt'
    var_2 = '/nonexistent/path'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None

def test_case_0():
    var_0 = 'Test find_hook returns None when no matching hooks are found.'
    var_1 = 'hooks'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns the absolute path when a valid hook is found.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = 'pre_prompt'
    var_5 = 0

def test_case_0():
    var_0 = 'Test find_hook ignores backup files (ending with ~).'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py~'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns multiple scripts with the same basename.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/usr/bin/env python\n'
    var_5 = '#!/bin/bash\n'
    var_6 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test find_hook uses default 'hooks' directory when not specified."
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = 'pre_prompt'
    var_5 = module_0.find_hook(var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #4
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
    var_0 = '/path/to/invalid-hook'
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
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/custom-hook'
    var_1 = 'custom-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 9/19 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 10/25 statements.


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
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda *args, **kwargs: var_7

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda *args, **kwargs: var_7

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with multiple pre_prompt scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test1'"
    var_5 = 493
    var_6 = 'pre_prompt.py'
    var_7 = "print('test2')"
    var_8 = 0
    assert var_8 == 2
    var_9 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 15/29 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 15/27 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 13/24 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/18 statements.
# Partially parsed test_run_script_os_error. Retrieved 6/18 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 16/31 statements.


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
    var_9 = []
    var_10 = 'subprocess.Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = len(var_9)
    assert var_14 == 1

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
    var_9 = []
    var_10 = 'subprocess.Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = len(var_9)
    assert var_14 == 1

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
    var_9 = 'subprocess.Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'test'
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = "print('success')"
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 0
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    assert var_15 == 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = 'pre-commit'
    var_3 = 'pre-commit'
    var_4 = 'pre-commit'
    var_5 = var_3 == var_4
    var_6 = True
    var_7 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'test_repo'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 6/12 statements.
# Partially parsed test_run_pre_prompt_hook_creates_tmp_dir_with_hook. Retrieved 12/30 statements.
# Partially parsed test_run_pre_prompt_hook_runs_script. Retrieved 12/29 statements.
# Partially parsed test_run_pre_prompt_hook_raises_failed_hook_exception. Retrieved 10/28 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_scripts. Retrieved 14/34 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = None
    var_5 = lambda name, hooks_dir='hooks': var_4

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temporary directory when pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 0
    var_7 = [var_6]
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = None
    var_11 = lambda script_path, cwd='.': var_10

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes the pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_10 = lambda d: Path(d)
    var_11 = len(var_6)

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_9 = lambda d: Path(d)

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook handles multiple pre_prompt scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test1'"
    var_5 = 493
    var_6 = 'pre_prompt.py'
    var_7 = "print('test2')"
    var_8 = []
    var_9 = 'cookiecutter.hooks.find_hook'
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_12 = lambda d: Path(d)
    var_13 = len(var_8)
    assert var_13 == 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.py'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_without_cleanup. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_0 = 'Test run_hook_from_repo_dir cleans up on FailedHookException when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up on FailedHookException when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on UndefinedError when delete_project_on_failure is True.'
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
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up on UndefinedError when delete_project_on_failure is False.'
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
    var_9 = 'os.getcwd'
    var_10 = 'os.chdir'
    var_11 = 'pre_prompt'
    var_12 = False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_script_path_ends_with_py. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'test_repo'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 3/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existent_hooks'
    var_1 = 'test_hook'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/19 statements.
# Partially parsed test_run_hook_single_script_found. Retrieved 13/20 statements.
# Partially parsed test_run_hook_multiple_scripts_found. Retrieved 14/22 statements.
# Partially parsed test_run_hook_with_string_project_dir. Retrieved 14/18 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 12/18 statements.


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
    var_1 = '/path/to/hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = [var_1]
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
    var_0 = 'Test run_hook when multiple hook scripts are found.'
    var_1 = '/path/to/hook1.sh'
    var_2 = '/path/to/hook2.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = [var_1, var_2]
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_gen_project'
    var_13 = 'Running hook %s'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook with string project_dir instead of Path.'
    var_1 = '/path/to/hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = [var_1]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = '/some/project/dir'
    var_12 = 'post_prompt'
    var_13 = module_0.run_hook(var_12, var_11, var_10)

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
    var_10 = 'pre_prompt'
    var_11 = 'No %s hook found'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 8/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/test'
    var_6 = 'pre_prompt'
    var_7 = module_0.run_hook(var_6, var_5, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 15/30 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 13/25 statements.
# Partially parsed test_run_script_with_context_renders_variables. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_with_pathlib_path. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders script with context and executes it.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = '_jinja2_env_vars'
    var_8 = 'test_project'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'myproject'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context properly renders Jinja2 variables.'
    var_1 = 'echo "{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}"'
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

def test_case_0():
    var_0 = 'Test run_script_with_context works with pathlib.Path objects.'
    var_1 = 'echo "{{ cookiecutter.msg }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'msg'
    var_6 = '_jinja2_env_vars'
    var_7 = 'hello'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 21 (except OSError) evaluates to False.'
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'make_executable'
    var_5 = '/path/to/script.sh'
    var_6 = '.'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_pre_prompt_hook_predicate_line_7_false. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (if not scripts:) evaluates to False.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook')"
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'non_existent_hooks'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/9 statements.
# Partially parsed test_work_in_returns_to_original_directory_on_exception. Retrieved 4/13 statements.
# Partially parsed test_work_in_with_none_stays_in_current_directory. Retrieved 1/5 statements.
# Partially parsed test_work_in_with_path_object. Retrieved 2/9 statements.
# Partially parsed test_work_in_with_string_path. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager changes to specified directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test that work_in with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test that work_in works with Path objects.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in works with string paths.'
    var_1 = 'test_subdir'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = 'No %s hook found'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_predicate_false. Retrieved 14/30 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is called with repo_dir argument.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'os'
    var_7 = __import__(var_6)
    var_8 = var_7.getcwd
    var_9 = __import__(var_6)
    var_10 = var_9.chdir
    var_11 = []
    var_12 = 'post_gen_project'
    var_13 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_exit_status_not_equal_to_success. Retrieved 3/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 9/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the delete parameter in NamedTemporaryFile is False.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = "echo 'test'"
    var_5 = '/tmp/script.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_3)
    var_8 = 1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 13/25 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 8/25 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 13/26 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 6/20 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'workdir'
    var_3 = {}
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_hook_returns_early_when_no_scripts_found. Retrieved 10/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when find_hook returns empty list.'
    var_1 = []
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = '/tmp/test'
    var_9 = module_0.run_hook(var_7, var_8, var_6)



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
    var_0 = 'pre-push'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid-hook'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-push'
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
    var_0 = '/path/to/pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py~'
    var_1 = 'pre-commit.py'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/commit-msg'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = module_0.valid_hook(var_1, var_0)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '/path/to/pre-commit'
    var_2 = module_0.valid_hook(var_1, var_0)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 3/9 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/18 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 6/19 statements.
# Partially parsed test_find_hook_uses_default_hooks_dir. Retrieved 6/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = '# dummy'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# dummy'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.py~'
    var_3 = '# dummy'
    var_4 = '# backup'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.sh'
    var_3 = '# python'
    var_4 = '# shell'
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# dummy'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 16/28 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 15/25 statements.
# Partially parsed test_run_script_with_context_python_extension. Retrieved 18/30 statements.
# Partially parsed test_run_script_with_context_preserves_cwd. Retrieved 14/27 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_extensions'
    var_7 = '_jinja2_env_vars'
    var_8 = 'my_project'
    var_9 = []
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = []
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = len(var_13)
    assert var_15 == 1

def test_case_0():
    var_0 = 'Test that run_script_with_context properly renders Jinja template.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.name }}'
    var_2 = 'render_test.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_extensions'
    var_7 = '_jinja2_env_vars'
    var_8 = 'test_value'
    var_9 = []
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = []
    var_14 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context with python script.'
    var_1 = "print('{{ cookiecutter.msg }}')"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'msg'
    var_6 = '_extensions'
    var_7 = '_jinja2_env_vars'
    var_8 = 'hello'
    var_9 = []
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = []
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = 0
    var_17 = var_13[var_16]

def test_case_0():
    var_0 = 'Test that run_script_with_context passes correct cwd to run_script.'
    var_1 = 'script.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'cookiecutter'
    var_6 = '_extensions'
    var_7 = '_jinja2_env_vars'
    var_8 = []
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/13 statements.
# Partially parsed test_run_hook_executes_found_scripts. Retrieved 11/23 statements.
# Partially parsed test_run_hook_single_script. Retrieved 10/17 statements.
# Partially parsed test_run_hook_with_pathlib_path. Retrieved 8/17 statements.
# Partially parsed test_run_hook_passes_context_correctly. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook executes found scripts.'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'pre_prompt'
    var_10 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook with a single script.'
    var_1 = 'post_gen_project.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook accepts pathlib.Path for project_dir.'
    var_1 = 'pre_prompt.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook passes context correctly to run_script_with_context.'
    var_1 = 'post_gen_project.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'my_project'
    var_8 = 'John Doe'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'post_gen_project'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 4/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_hook'
    var_2 = 'nonexistent_hooks_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 13/26 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir succeeds when hook runs successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'post_gen_project'
    var_11 = False
    var_12 = len(var_8)
    assert var_12 == 1

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
    var_9 = 'post_gen_project'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir keeps project when delete_project_on_failure is False.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'post_gen_project'
    var_11 = False
    var_12 = len(var_8)
    assert var_12 == 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'pre-commit'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'pre_prompt'
    var_6 = '/tmp/test'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = module_0.run_hook(var_5, var_6, var_9)
    var_11 = 'No %s hook found'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 7/19 statements.
# Partially parsed test_run_pre_prompt_hook_returns_temp_dir. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt script exists.'
    var_1 = 'repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt script.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a temporary directory path.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 9/29 statements.


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
    var_7 = 'test_hook'
    var_8 = all(var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/15 statements.
# Partially parsed test_run_hook_executes_single_script. Retrieved 16/22 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 17/25 statements.
# Partially parsed test_run_hook_passes_context_to_scripts. Retrieved 18/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = lambda hook_name: var_2
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes a single hook script.'
    var_1 = None
    var_2 = lambda script_path, project_dir, context: var_1
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = '/hooks/post_gen_project.sh'
    var_6 = [var_5]
    var_7 = lambda hook_name: var_6
    var_8 = 'post_gen_project'
    var_9 = '/project'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = module_0.run_hook(var_8, var_9, var_14)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = []
    var_2 = 'cookiecutter.hooks.run_script_with_context'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = '/hooks/script1.sh'
    var_5 = '/hooks/script2.py'
    var_6 = [var_4, var_5]
    var_7 = lambda hook_name: var_6
    var_8 = 'pre_gen_project'
    var_9 = '/project'
    var_10 = 'cookiecutter'
    var_11 = 'name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = module_0.run_hook(var_8, var_9, var_14)
    var_16 = len(var_1)
    assert var_16 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook passes context to run_script_with_context.'
    var_1 = []
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'myproject'
    var_6 = 'me'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter.hooks.run_script_with_context'
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = '/hooks/post_gen.sh'
    var_12 = [var_11]
    var_13 = lambda hook_name: var_12
    var_14 = 'post_gen_project'
    var_15 = '/my/project'
    var_16 = module_0.run_hook(var_14, var_15, var_8)
    var_17 = len(var_1)
    assert var_17 == 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 12/36 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_prompt'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.py'
    var_7 = '# hook script'
    var_8 = 'pre_prompt'
    var_9 = len(var_2)
    var_10 = 'hooks'
    var_11 = 'any_hook'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'pre_prompt'
    var_4 = []
    var_5 = None
    var_6 = lambda hook_name: var_4 if hook_name == var_3 else var_5



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/19 statements.
# Partially parsed test_run_hook_single_script. Retrieved 12/22 statements.
# Partially parsed test_run_hook_multiple_scripts. Retrieved 13/26 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 12/19 statements.
# Partially parsed test_run_hook_with_path_object. Retrieved 13/21 statements.


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
    var_0 = 'Test run_hook with a single hook script.'
    var_1 = 'hook.py'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'
    var_11 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook with multiple hook scripts.'
    var_1 = 'hook1.py'
    var_2 = 'hook2.sh'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_gen_project'
    var_12 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns empty list.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
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
    var_0 = 'Test run_hook with Path object as project_dir.'
    var_1 = '/tmp/hook.py'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = [var_1]
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 4/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_hook'
    var_2 = 'non_existent_hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'Test that find_hook returns None when hooks directory does not exist.'
    var_1 = 'non_existent_hooks'
    var_2 = 'test_hook'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 15/30 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 14/27 statements.
# Partially parsed test_run_script_failed_hook_non_zero_exit. Retrieved 13/27 statements.
# Partially parsed test_run_script_failed_hook_enoexec. Retrieved 8/25 statements.
# Partially parsed test_run_script_failed_hook_os_error. Retrieved 9/25 statements.


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
    var_9 = 'Popen'
    var_10 = 'sys.executable'
    var_11 = 'python'
    var_12 = None
    var_13 = lambda path: var_12
    var_14 = 'make_executable'

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
    var_9 = 'Popen'
    var_10 = None
    var_11 = lambda path: var_10
    var_12 = 'make_executable'
    var_13 = '.'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = None
    var_11 = lambda path: var_10
    var_12 = 'make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = None
    var_3 = lambda path: var_2
    var_4 = 'make_executable'
    var_5 = OSError()
    var_6 = ()
    var_7 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = None
    var_3 = lambda path: var_2
    var_4 = 'make_executable'
    var_5 = 'Permission denied'
    var_6 = OSError(var_5)
    var_7 = ()
    var_8 = 'Popen'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 11/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile.NamedTemporaryFile is called with delete=False at line 14.'
    var_1 = None
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'echo "test"'
    var_6 = 'rendered output'
    var_7 = '/path/to/script.sh'
    var_8 = '/cwd'
    var_9 = module_0.run_script_with_context(var_7, var_8, var_4)
    var_10 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/4 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/9 statements.
# Partially parsed test_find_hook_returns_absolute_path_for_matching_hook. Retrieved 5/12 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/13 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 6/13 statements.
# Partially parsed test_find_hook_with_custom_hooks_dir. Retrieved 5/12 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = 'pre_prompt'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt'
    var_3 = module_0.find_hook(var_2, var_0)
    var_4 = len(var_3)
    assert var_4 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.sh~'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt'
    var_3 = module_0.find_hook(var_2, var_0)
    var_4 = len(var_3)
    assert var_4 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.sh'
    var_2 = 'unsupported_hook'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 17/29 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 17/29 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 17/29 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_no_delete. Retrieved 17/29 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.run_hook'
    var_3 = 'cookiecutter.hooks.rmtree'
    var_4 = '/path/to/repo'
    var_5 = 'post_gen_project'
    var_6 = '/path/to/project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = None
    var_13 = False
    var_14 = module_0.run_hook_from_repo_dir(var_4, var_5, var_6, var_11, var_13)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException and deletes project.'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.run_hook'
    var_3 = 'cookiecutter.hooks.rmtree'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = '/path/to/repo'
    var_6 = 'post_gen_project'
    var_7 = '/path/to/project'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = None
    var_14 = 'Hook failed'
    var_15 = True
    var_16 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_12, var_15)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException without deleting project.'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.run_hook'
    var_3 = 'cookiecutter.hooks.rmtree'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = '/path/to/repo'
    var_6 = 'pre_prompt'
    var_7 = '/path/to/project'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = None
    var_14 = 'Hook failed'
    var_15 = False
    var_16 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_12, var_15)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError and deletes project.'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.run_hook'
    var_3 = 'cookiecutter.hooks.rmtree'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = '/path/to/repo'
    var_6 = 'post_gen_project'
    var_7 = '/path/to/project'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = None
    var_14 = 'Undefined variable'
    var_15 = True
    var_16 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_12, var_15)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError without deleting project.'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.run_hook'
    var_3 = 'cookiecutter.hooks.rmtree'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = '/path/to/repo'
    var_6 = 'pre_prompt'
    var_7 = '/path/to/project'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = None
    var_14 = 'Undefined variable'
    var_15 = False
    var_16 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_12, var_15)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/24 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 7/22 statements.
# Partially parsed test_run_script_windows_uses_shell. Retrieved 7/22 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 5/19 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/22 statements.
# Partially parsed test_run_script_other_oserror. Retrieved 5/19 statements.


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

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = len(var_2)
    assert var_6 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid'
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/20 statements.
# Partially parsed test_run_hook_from_repo_dir_restores_working_directory_on_exception. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir succeeds when hook runs successfully.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't delete project on FailedHookException when flag is False."
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = False

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
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

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
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir restores original working directory even on exception.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'pre_prompt'
    var_13 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 14/36 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = '#!/bin/bash\n'
    var_5 = 'test_hook'
    var_6 = 'hooks'
    var_7 = 'test_hook.sh'
    var_8 = '#!/bin/bash\n'
    var_9 = 'test_hook'
    var_10 = len(var_2)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_2[var_11]
    var_13 = var_2[var_11]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 9/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_0, var_6, var_2, var_5, var_7)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_jinja2_env_vars'
    var_4 = 'test_project'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = "#!/bin/bash\necho 'test'"
    var_9 = '/tmp/test_script.sh'
    var_10 = '/tmp'
    var_11 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 14 (delete=False) evaluates to False.'
    var_1 = 'echo "test"'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    assert var_5 is False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 10/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_prompt'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.sh'
    var_7 = '#!/bin/bash\necho "test"'
    var_8 = 'pre_prompt'
    var_9 = len(var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('test')"
    var_3 = 'pre_prompt'
    var_4 = 'pre_prompt.py'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 6/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Popen'
    var_1 = 'make_executable'
    var_2 = None
    var_3 = lambda x: var_2
    var_4 = '/path/to/script.sh'
    var_5 = module_0.run_script(var_4)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with delete=False and correct suffix.'
    var_1 = '/path/to/script.sh'
    var_2 = '/working/directory'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.run_script_with_context(var_1, var_2, var_7)
    var_9 = 1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_exit_status_not_equal_to_exit_success. Retrieved 3/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('test')"
    var_3 = 'pre_prompt'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_is_empty. Retrieved 1/4 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hook. Retrieved 3/9 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_found. Retrieved 3/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/9 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 4/16 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = '/nonexistent/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_prompt'

def test_case_0():
    var_0 = 'post_gen_project.py'
    var_1 = 'w'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt.py'
    var_1 = 'w'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt.py~'
    var_1 = 'w'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt.py'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt.py'
    var_1 = 'w'
    var_2 = 'pre_prompt'
    var_3 = 0



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 11/29 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None
    var_2 = 'test_hook'
    var_3 = 'hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None
    var_5 = 'hook1.sh'
    var_6 = 'hook2.sh'
    var_7 = 'test_hook'
    var_8 = 'hooks'
    var_9 = module_0.find_hook(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 2



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = var_1 != var_0
    assert var_2 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/8 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 4/11 statements.
# Partially parsed test_find_hook_matching_script. Retrieved 7/17 statements.
# Partially parsed test_find_hook_backup_file_ignored. Retrieved 6/15 statements.
# Partially parsed test_find_hook_unsupported_hook_ignored. Retrieved 6/15 statements.
# Partially parsed test_find_hook_multiple_matching_scripts. Retrieved 9/22 statements.
# Partially parsed test_find_hook_custom_hooks_dir. Retrieved 7/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'unsupported_hook'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.sh'
    var_3 = '#!/usr/bin/env python'
    var_4 = '#!/bin/bash'
    var_5 = 'pre_prompt'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'custom_hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false_no_delete. Retrieved 14/30 statements.


def test_case_0():
    var_0 = 'Test that project_dir is not deleted when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = None
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'test error'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'pre_prompt'
    var_13 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 6/16 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 6/14 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 6/14 statements.
# Partially parsed test_find_hook_with_nonexistent_hooks_dir. Retrieved 2/6 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 8/19 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'post_gen_project'
    var_5 = [var_1, var_4]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt~'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = "#!/bin/bash\necho 'test1'"
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('test2')"
    var_5 = '__main__._HOOKS'
    var_6 = 'pre_prompt'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 4/9 statements.
# Partially parsed test_run_pre_prompt_hook_executes_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_raises_on_failed_script. Retrieved 7/19 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_copy. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.json'
    var_3 = '{}'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script when it exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary copy of repo_dir.'
    var_1 = 'test_repo'
    var_2 = 'test_file.txt'
    var_3 = 'content'
    var_4 = 'hooks'
    var_5 = 'pre_prompt.sh'
    var_6 = "#!/bin/bash\necho 'test'"
    var_7 = 493
    var_8 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_preserves_project_on_success. Retrieved 9/21 statements.


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

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir preserves project directory when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_found. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 8/17 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_exception. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 10/23 statements.
# Partially parsed test_run_pre_prompt_hook_returns_temp_dir. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid pre_prompt hook.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('Hook executed')"
    var_5 = None
    var_6 = lambda script_path, cwd='.': var_5
    var_7 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when hook fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('Hook executed')"
    var_5 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with multiple hook scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('Hook 1')"
    var_5 = 'pre_prompt.sh'
    var_6 = "#!/bin/bash\necho 'Hook 2'"
    var_7 = 0
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a different temporary directory.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('Hook')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = None
    var_7 = lambda script_path, cwd='.': var_6



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'post_gen_project.py'
    var_5 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 13/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_delete_project. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_keep_project. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_delete_project. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_keep_project. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 15/27 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir succeeds when hook runs successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'pre_prompt'
    var_11 = False
    var_12 = len(var_8)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir keeps project on FailedHookException when flag is False.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir keeps project on UndefinedError when flag is False.'
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
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'os'
    var_11 = __import__(var_10)
    var_12 = 'pre_prompt'
    var_13 = False
    var_14 = __import__(var_10)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_logs_exception. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_0 = 'Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True.'
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
    var_12 = 'pre_prompt'
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
    var_8 = None
    var_9 = None
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'post_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir logs exception when hook fails.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 7/25 statements.


def test_case_0():
    var_0 = "Test that the predicate 'delete=False' at line 14 evaluates to False."
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'echo "test"'
    var_5 = []
    var_6 = '.'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when pre_prompt script is not found.'
    var_1 = 'test_repo'
    var_2 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 9/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/test.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Exec format error'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 10/35 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = 'temp_repo'
    var_2 = 'temp_project'
    var_3 = True
    var_4 = None
    var_5 = 'post_gen_project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = False



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_oserror_predicate_evaluates_to_false. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Some other error'
    var_1 = OSError(var_0)
    var_2 = '/path/to/script.sh'
    var_3 = module_0.run_script(var_2)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_exit_status_equals_success. Retrieved 4/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = '/path/to/script.sh'
    var_2 = '.'
    var_3 = module_0.run_script(var_1, var_2)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 20 evaluates to False when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = []
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = len(var_7)
    assert var_11 == 0



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_exit_status_equals_exit_success. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = 0
    var_2 = '/path/to/script.sh'
    var_3 = module_0.run_script(var_2)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_without_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter.hooks.rmtree'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_prompt'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir when executing hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_prompt'
    var_11 = False
    var_12 = str(var_3)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 10/26 statements.


def test_case_0():
    var_0 = 'Test that tempfile is created with delete=False parameter.'
    var_1 = '/tmp/test_script.sh'
    var_2 = '/tmp'
    var_3 = 'cookiecutter'
    var_4 = 'test_var'
    var_5 = 'test_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = None
    var_9 = 1



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 9/25 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 8/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 7/19 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 7/20 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 7/17 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = len(var_2)
    assert var_8 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'hello'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = module_0.run_script(var_0)

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'workdir'
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = len(var_3)
    assert var_8 == 1



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_run_pre_prompt_hook_predicate_false. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 9 (if not scripts) evaluates to False.'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'pre_prompt'
    var_7 = False
    var_8 = 'pre_prompt'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/20 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_with_deletion. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_without_deletion. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/27 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = len(var_3)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'post_gen_project'
    var_8 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir keeps project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes from repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = len(var_3)
    assert var_10 == 1
    var_11 = var_3[var_9]
    var_12 = var_3[var_9]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = '/some/repo/dir'
    var_2 = '/some/project/dir'
    var_3 = 'post_gen_project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_restores_working_directory. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_restores_working_directory_on_exception. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = True

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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = False

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
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir restores original working directory after execution.'
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
    var_0 = 'Test run_hook_from_repo_dir restores working directory even on exception.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_found. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 6/15 statements.
# Partially parsed test_run_pre_prompt_hook_python_script. Retrieved 6/16 statements.
# Partially parsed test_run_pre_prompt_hook_returns_new_path. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temp dir and runs pre_prompt script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes Python pre_prompt script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns path to temporary directory copy.'
    var_1 = 'template'
    var_2 = 'cookiecutter.json'
    var_3 = '{}'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 8/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 8/24 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 6/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/21 statements.
# Partially parsed test_run_script_os_error. Retrieved 6/18 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')\n"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'\n"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys\nsys.exit(1)\n'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')\n"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'custom'
    var_2 = "print('test')\n"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')\n"
    var_2 = 'Popen'
    var_3 = False
    var_4 = True
    assert var_4 is False



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_run_script_python_file. Retrieved 4/16 statements.
# Partially parsed test_run_script_non_python_file. Retrieved 3/9 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 2/8 statements.
# Partially parsed test_run_script_with_path_object. Retrieved 2/11 statements.


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
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = module_0.run_script(var_0)

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/tmp'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 21 (err.errno == errno.ENOEXEC) evaluates to False.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')\n"
    var_3 = 'Some other error'
    var_4 = OSError(var_3)
    var_5 = 'subprocess.Popen'
    var_6 = 'utils.make_executable'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




# Parsed testcases at query #76
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []
    var_4 = lambda hook_name: var_3



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/17 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 11/17 statements.
# Partially parsed test_run_hook_executes_single_script. Retrieved 12/23 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 12/25 statements.
# Partially parsed test_run_hook_with_pathlib_path. Retrieved 13/21 statements.
# Partially parsed test_run_hook_passes_context_to_script. Retrieved 15/24 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'
    var_10 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook when scripts list is empty.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'
    var_10 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook executes a single script.'
    var_1 = 'test_hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'
    var_11 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook executes multiple scripts in order.'
    var_1 = 'hook1.sh'
    var_2 = 'hook2.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook accepts pathlib.Path for project_dir.'
    var_1 = '/tmp/test_hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = [var_1]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = '/tmp/project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook passes the context to the script.'
    var_1 = 'test_hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'author'
    var_8 = 'my_project'
    var_9 = 'John'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 'pre_prompt'
    var_13 = 2
    var_14 = 0



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 12/22 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 12/21 statements.
# Partially parsed test_run_script_python_file_failure. Retrieved 12/22 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 7/20 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 8/21 statements.
# Partially parsed test_run_script_uses_shell_on_windows. Retrieved 8/20 statements.
# Partially parsed test_run_script_command_format_python. Retrieved 6/17 statements.
# Partially parsed test_run_script_command_format_non_python. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = OSError()
    var_2 = ()
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Permission denied'
    var_2 = OSError(var_1)
    var_3 = ()
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'sys.platform'
    var_7 = 'win32'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/6 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_script_when_matching_hook_exists. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 5/18 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_uses_default_hooks_dir. Retrieved 6/14 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/12 statements.


def test_case_0():
    var_0 = "Test find_hook returns None when hooks directory doesn't exist."
    var_1 = 'pre_prompt'
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = 'Test find_hook returns None when no matching hooks are found.'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.sh'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns absolute path when matching hook exists.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores backup files ending with ~.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh~'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns multiple scripts with the same hook name.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt.py'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores hooks that are not in _HOOKS.'
    var_1 = 'hooks'
    var_2 = 'unsupported_hook.sh'
    var_3 = 'unsupported_hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test find_hook uses default 'hooks' directory."
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = 'Test find_hook returns absolute paths.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_scripts_list_when_matching_hook_exists. Retrieved 9/21 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 10/24 statements.
# Partially parsed test_find_hook_returns_multiple_matching_scripts. Retrieved 9/21 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'non_existent_hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'builtins.__import__'
    var_4 = 'cookiecutter.hooks._HOOKS'
    var_5 = 'pre_prompt'
    var_6 = 'post_gen_project'
    var_7 = [var_5, var_6]
    var_8 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt.sh~'
    var_4 = 'cookiecutter.hooks._HOOKS'
    var_5 = 'pre_prompt'
    var_6 = [var_5]
    var_7 = 'pre_prompt'
    var_8 = 0
    var_9 = '~'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\n'
    var_5 = 'cookiecutter.hooks._HOOKS'
    var_6 = 'pre_prompt'
    var_7 = [var_6]
    var_8 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'cookiecutter.hooks._HOOKS'
    var_4 = 'post_gen_project'
    var_5 = [var_4]
    var_6 = 'post_gen_project'
    var_7 = module_0.find_hook(var_6)
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 10/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = '/test/repo'
    var_2 = '/test/project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_7, var_2, var_5, var_8)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir changes to repo_dir using work_in context manager.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = str(var_2)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_cleanup. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_directory. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined variable'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_gen_project'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = None
    var_5 = 'cookiecutter.hooks.run_hook'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'
    var_12 = False



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/30 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 13/29 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 11/26 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 4/21 statements.
# Partially parsed test_run_script_oserror. Retrieved 4/18 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 14/32 statements.


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
    var_9 = []
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = len(var_9)
    assert var_12 == 1

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
    var_9 = []
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = len(var_9)
    assert var_12 == 1

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
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = "print('test')"
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 0
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = []
    var_11 = 'Popen'
    var_12 = 'utils.make_executable'
    var_13 = len(var_10)
    assert var_13 == 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/6 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 9/23 statements.
# Partially parsed test_run_pre_prompt_hook_hook_script_fails. Retrieved 10/26 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_hook_scripts. Retrieved 11/28 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when a valid pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = None

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when hook script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = None
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 'Script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with multiple hook scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook 1')"
    var_5 = 'pre_prompt.sh'
    var_6 = "#!/bin/bash\necho 'hook 2'"
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_9 = 'cookiecutter.hooks.find_hook'
    var_10 = None



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
    var_0 = '/some/nested/path/commit-msg'
    var_1 = 'commit-msg'
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
    var_0 = '/path/to/pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.sh~'
    var_1 = 'pre-commit.sh'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/.bashrc'
    var_1 = '.bashrc'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/prepare-commit-msg'
    var_1 = 'prepare-commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/8 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 4/11 statements.
# Partially parsed test_find_hook_matching_hook_found. Retrieved 7/17 statements.
# Partially parsed test_find_hook_backup_file_ignored. Retrieved 6/15 statements.
# Partially parsed test_find_hook_unsupported_hook_ignored. Retrieved 6/15 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 9/22 statements.
# Partially parsed test_find_hook_custom_hooks_dir. Retrieved 7/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'unsupported_hook'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash'
    var_5 = 'pre_prompt'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 'custom_hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_hook'
    var_1 = 'test_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_no_cleanup_on_failure. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/21 statements.


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
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException and cleans up.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError and cleans up.'
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

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir while executing.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_hook_predicate_line_15_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/19 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 13/20 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 18/26 statements.
# Partially parsed test_run_hook_with_empty_scripts_list. Retrieved 12/19 statements.
# Partially parsed test_run_hook_passes_correct_hook_name. Retrieved 8/12 statements.


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
    var_0 = 'Test run_hook when a single hook script is found and executed.'
    var_1 = '/path/to/hook.py'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = [var_1]
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
    var_0 = 'Test run_hook when multiple hook scripts are found and all are executed.'
    var_1 = '/path/to/hook1.py'
    var_2 = '/path/to/hook2.sh'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_gen_project'
    var_13 = 0
    var_14 = var_3[var_13]
    var_15 = 1
    var_16 = var_3[var_15]
    var_17 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook when an empty scripts list is returned.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_prompt'
    var_11 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test that run_hook passes the correct hook_name to find_hook.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'custom_hook'



# Parsed testcases at query #10
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
    var_8 = 'pre_prompt'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_with_failed_hook. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 9/28 statements.


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
    var_0 = 'Test run_pre_prompt_hook when hook script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'test.txt'
    var_7 = 'test content'
    var_8 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'pre-commit'
    var_1 = '#!/bin/bash\n'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/21 statements.


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
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
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
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = 'pre_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir while running hook.'
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_absolute_path_for_matching_hook. Retrieved 5/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_multiple_matching_scripts. Retrieved 6/17 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/12 statements.
# Partially parsed test_find_hook_returns_none_for_unsupported_hook. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent'

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
    var_4 = 0

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
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.sh'
    var_2 = '#!/bin/bash'
    var_3 = 'unsupported_hook'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_pre_prompt_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_pre_prompt_script. Retrieved 5/15 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_script. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_directory. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt script exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script successfully.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('pre_prompt executed')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when pre_prompt script exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'test.txt'
    var_4 = 'test content'
    var_5 = 'pre_prompt.py'
    var_6 = "print('running')"



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_with_extensions. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 13/22 statements.
# Partially parsed test_run_script_with_context_cwd_parameter. Retrieved 14/26 statements.
# Partially parsed test_run_script_with_context_multiple_variables. Retrieved 15/25 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.project_name }}')"
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'my_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = 0
    var_12 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context with Jinja2 extensions.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.message | slugify }}'"
    var_3 = 'cookiecutter'
    var_4 = 'message'
    var_5 = '_jinja2_env_vars'
    var_6 = '_extensions'
    var_7 = 'Hello World'
    var_8 = {}
    var_9 = []
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = 0
    var_14 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = 'hook.sh'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.name }}'"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'test'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = 0
    var_12 = '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context passes cwd to run_script.'
    var_1 = 'script.py'
    var_2 = "print('{{ cookiecutter.value }}')"
    var_3 = 'workdir'
    var_4 = 'cookiecutter'
    var_5 = 'value'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_value'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = 1
    var_13 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = 'test.py'
    var_2 = "print('{{ cookiecutter.name }}-{{ cookiecutter.version }}')"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'version'
    var_6 = '_jinja2_env_vars'
    var_7 = 'myapp'
    var_8 = '1.0.0'
    var_9 = {}
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = 0
    var_14 = 'utf-8'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda x: var_2
    var_4 = 'pre_prompt'
    var_5 = '.'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre-commit'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_with_extension. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 11/21 statements.
# Partially parsed test_run_script_with_context_passes_cwd. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.name }}')"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'test_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = 0
    var_12 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context with jinja2 extensions.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.message | slugify }}'"
    var_3 = 'cookiecutter'
    var_4 = 'message'
    var_5 = '_jinja2_env_vars'
    var_6 = '_extensions'
    var_7 = 'Hello World'
    var_8 = {}
    var_9 = []
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = 0
    var_14 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context creates temporary file with correct extension.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'cookiecutter'
    var_4 = '_jinja2_env_vars'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 0
    var_10 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct working directory.'
    var_1 = 'test_script.sh'
    var_2 = 'pwd'
    var_3 = 'cookiecutter'
    var_4 = '_jinja2_env_vars'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 1
    var_10 = 0



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 10/31 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'test_hook'
    var_5 = 'hooks'
    var_6 = 'test_hook.sh'
    var_7 = '#!/bin/bash\n'
    var_8 = 'test_hook'
    var_9 = len(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = None
    var_8 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_9 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = 'No %s hook found'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 6/16 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 2/6 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 7/15 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 7/15 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 9/20 statements.
# Partially parsed test_find_hook_with_non_matching_hook_name. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'post_gen_project'
    var_5 = [var_1, var_4]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt~'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = '__main__._HOOKS'
    var_6 = 'pre_prompt'
    var_7 = 'post_gen_project'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'post_gen_project'
    var_5 = [var_1, var_4]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_valid_hook_returns_true_for_matching_supported_non_backup_hook. Retrieved 5/17 statements.


def test_case_0():
    var_0 = '__main__._HOOKS'
    var_1 = 'test_hook'
    var_2 = {var_1}
    var_3 = "#!/bin/bash\necho 'test'"
    var_4 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used at line 17.'
    var_1 = 'test_project'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'post_gen_project'
    var_6 = None
    var_7 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 2/8 statements.
# Partially parsed test_find_hook_returns_list_with_single_hook. Retrieved 7/19 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/17 statements.
# Partially parsed test_find_hook_returns_multiple_hooks. Retrieved 8/23 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 7/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'post_gen_project'
    var_2 = [var_0, var_1]
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = '#!/usr/bin/env python\n'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py~'
    var_4 = '#!/usr/bin/env python\n'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/usr/bin/env python\n'
    var_6 = '#!/bin/bash\n'
    var_7 = 'pre_prompt'

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'post_gen_project.py'
    var_4 = '#!/usr/bin/env python\n'
    var_5 = 'post_gen_project'
    var_6 = 0



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 4/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_python_hook. Retrieved 6/15 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_bash_hook. Retrieved 6/15 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_raises_exception. Retrieved 6/16 statements.
# Partially parsed test_run_pre_prompt_hook_string_repo_dir. Retrieved 4/11 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_copy. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_hooks. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'template'
    var_2 = 'cookiecutter.json'
    var_3 = '{}'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid Python pre_prompt hook.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "# Valid hook\nprint('Hook executed')"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid bash pre_prompt hook.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'Hook executed'"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when hook fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys\nsys.exit(1)'
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook works with string repo_dir parameter.'
    var_1 = 'template'
    var_2 = 'cookiecutter.json'
    var_3 = '{}'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary copy of repo_dir.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# Valid hook'
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt hooks.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# Hook 1'
    var_5 = 493
    var_6 = 'pre_prompt.sh'
    var_7 = '#!/bin/bash\n# Hook 2'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 11/25 statements.
# Partially parsed test_run_script_with_context_with_jinja_variables. Retrieved 11/22 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 14/25 statements.
# Partially parsed test_run_script_with_context_multiple_variables. Retrieved 15/26 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')\n"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context properly renders Jinja2 variables.'
    var_1 = 'script.sh'
    var_2 = '#!/bin/bash\necho {{ cookiecutter.version }}\n'
    var_3 = 'cookiecutter'
    var_4 = 'version'
    var_5 = '1.0.0'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = 'script.py'
    var_2 = "print('{{ cookiecutter.msg }}')\n"
    var_3 = 'cookiecutter'
    var_4 = 'msg'
    var_5 = 'hello'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8[var_11]
    var_13 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = 'script.sh'
    var_2 = '{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}-{{ cookiecutter.var3 }}\n'
    var_3 = 'cookiecutter'
    var_4 = 'var1'
    var_5 = 'var2'
    var_6 = 'var3'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/20 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 4/18 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 4/19 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 4/19 statements.
# Partially parsed test_run_script_oserror. Retrieved 4/16 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script'
    var_1 = ''
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'custom'
    var_3 = []
    var_4 = 'subprocess.Popen'
    var_5 = 'utils.make_executable'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 8/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = 'find_hook.valid_hook'
    var_4 = 'post_gen_project'
    var_5 = lambda hook_file, hook_name: hook_file == var_1 and hook_name == var_4
    var_6 = module_0.find_hook(var_4, var_0)
    var_7 = len(var_6)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/31 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 13/29 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 13/28 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 12/26 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'builtins.__import__'
    var_5 = 'utils'
    var_6 = ()
    var_7 = 'make_executable'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = len(var_2)
    assert var_12 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'builtins.__import__'
    var_5 = 'utils'
    var_6 = ()
    var_7 = 'make_executable'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = len(var_2)
    assert var_12 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = 'builtins.__import__'
    var_4 = 'utils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'builtins.__import__'
    var_3 = 'utils'
    var_4 = ()
    var_5 = 'make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = False
    var_11 = True

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'builtins.__import__'
    var_3 = 'utils'
    var_4 = ()
    var_5 = 'make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = False
    var_11 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 5/20 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 5/19 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 3/16 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/16 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 3/13 statements.
# Partially parsed test_run_script_windows_shell. Retrieved 6/20 statements.
# Partially parsed test_run_script_custom_cwd. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')\n"
    var_2 = []
    var_3 = 'Popen'
    var_4 = len(var_2)
    assert var_4 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'\n"
    var_2 = []
    var_3 = 'Popen'
    var_4 = len(var_2)
    assert var_4 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)\n'
    var_2 = 'Popen'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')\n"
    var_2 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')\n"
    var_2 = []
    var_3 = 'platform'
    var_4 = 'win32'
    var_5 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = "print('test')\n"
    var_3 = []
    var_4 = 'Popen'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/25 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 13/24 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 13/24 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 6/18 statements.
# Partially parsed test_run_script_os_error_generic. Retrieved 6/17 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'obj'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda : var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = 'obj'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda : var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'obj'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = lambda : var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'Popen'
    var_10 = 'utils.make_executable'
    var_11 = None
    var_12 = lambda x: var_11

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'workdir'
    var_3 = {}
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used correctly in run_pre_prompt_hook.'
    var_1 = 'pre_prompt_script.py'
    var_2 = [var_1]
    var_3 = None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test that project directory is not deleted when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'post_gen_project'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project.py'
    var_7 = False



# Parsed testcases at query #37
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 14/25 statements.
# Partially parsed test_run_script_with_context_with_multiple_variables. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 16/27 statements.
# Partially parsed test_run_script_with_context_uses_correct_cwd. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_with_jinja_filters. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.name }}')"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'test_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 0
    var_11 = [var_10]
    var_12 = None
    var_13 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple template variables.'
    var_1 = 'script.sh'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'\necho '{{ cookiecutter.author }}'"
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = 'John Doe'
    var_9 = {}
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = 'hook.py'
    var_2 = "print('{{ cookiecutter.value }}')"
    var_3 = 'cookiecutter'
    var_4 = 'value'
    var_5 = '_jinja2_env_vars'
    var_6 = 'test_value'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_10[var_13]
    var_15 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct working directory.'
    var_1 = 'script.py'
    var_2 = "echo '{{ cookiecutter.name }}'"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with Jinja2 filters in template.'
    var_1 = 'script.py'
    var_2 = "echo '{{ cookiecutter.name|upper }}'"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'lowercase'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 8/23 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    assert var_1 is None
    var_2 = None
    var_3 = None
    assert var_3 is None
    var_4 = '/abs/path/hook1'
    var_5 = '/abs/path/hook2'
    var_6 = [var_4, var_5]
    var_7 = None



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_exit_status_equals_success. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.py'
    var_2 = '.'
    var_3 = module_0.run_script(var_1, var_2)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_script_success_with_zero_exit_status. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'Test that predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = 'Popen'
    var_2 = 'make_executable'
    var_3 = 'test_script.py'
    var_4 = "print('test')"



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_line_18_evaluates_to_true. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'test_script.sh'
    var_2 = [var_1]
    var_3 = False
    var_4 = '.'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 13/32 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 12/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 12/23 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/18 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'MockProc'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 1
    var_6 = True
    var_7 = 0
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = len(var_1)

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'MockProc'
    var_2 = ()
    var_3 = 'wait'
    var_4 = 1
    var_5 = lambda self: var_4
    var_6 = {var_3: var_5}
    var_7 = type(var_1, var_2, var_6)
    var_8 = 'Popen'
    var_9 = 'utils.make_executable'
    var_10 = None
    var_11 = lambda x: var_10

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/24 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_windows_shell. Retrieved 6/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/22 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'hello'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/23 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 12/22 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 15/25 statements.
# Partially parsed test_run_script_with_context_with_jinja_env_vars. Retrieved 14/23 statements.
# Partially parsed test_run_script_with_context_uses_correct_cwd. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('{{ cookiecutter.name }}')"
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_jinja2_env_vars'
    var_5 = 'test_project'
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_jinja2_env_vars'
    var_5 = 'my_awesome_project'
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1

def test_case_0():
    var_0 = 'test_script.rb'
    var_1 = "puts '{{ cookiecutter.version }}'"
    var_2 = 'cookiecutter'
    var_3 = 'version'
    var_4 = '_jinja2_env_vars'
    var_5 = '1.0.0'
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1
    var_12 = 0
    var_13 = var_9[var_12]
    var_14 = '.rb'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('{% if true %}hello{% endif %}')"
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = '_extensions'
    var_5 = 'trim_blocks'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = []
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '# test'
    var_2 = 'custom_dir'
    var_3 = 'cookiecutter'
    var_4 = '_jinja2_env_vars'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 6/25 statements.
# Partially parsed test_find_hook_returns_none_when_no_hooks_dir. Retrieved 3/10 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = None
    var_5 = type(var_4)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 10/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = '/fake/repo'
    var_2 = '/fake/project'
    var_3 = 'post_gen_project.py'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_8)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = None
    var_8 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_9 = 1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_oserror_enoexec_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without errors.'
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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = False

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
    var_12 = 'pre_prompt'
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
    var_8 = []
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True

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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'post_gen_project'
    var_11 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 9/30 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test that tempfile is created with delete=False, mode='wb', and correct suffix."
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/path/to/script.sh'
    var_6 = '/cwd'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 12/22 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter.hooks.rmtree'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'pre_prompt'
    var_10 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Variable undefined'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_prompt'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 11/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 11/25 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 10/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 10/22 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/16 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'obj'
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda : var_4
    var_6 = {var_3: var_5}
    var_7 = 'Popen'
    var_8 = 'utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'obj'
    var_3 = 'wait'
    var_4 = 0
    var_5 = lambda : var_4
    var_6 = {var_3: var_5}
    var_7 = 'Popen'
    var_8 = 'utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'obj'
    var_2 = 'wait'
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = {var_2: var_4}
    var_6 = 'Popen'
    var_7 = 'utils.make_executable'
    var_8 = None
    var_9 = lambda x: var_8

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'obj'
    var_2 = 'wait'
    var_3 = 1
    var_4 = lambda : var_3
    var_5 = {var_2: var_4}
    var_6 = 'Popen'
    var_7 = 'utils.make_executable'
    var_8 = None
    var_9 = lambda x: var_8

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Hook failed'
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Hook failed'
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Undefined variable'
    var_9 = module_0.UndefinedError(var_8)
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'cookiecutter.hooks.logger'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = None
    var_8 = None
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = False



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 5/21 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 3/16 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/20 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 3/17 statements.
# Partially parsed test_run_script_oserror. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 'Popen'
    var_3 = 'sys.executable'
    var_4 = 'python'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'Popen'
    var_3 = 'sys.executable'
    var_4 = 'python'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/23 statements.


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
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
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
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before executing hook.'
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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir successfully runs a hook.'
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
    var_0 = 'Test run_hook_from_repo_dir cleans up project on FailedHookException.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up project when flag is False.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on UndefinedError.'
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
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'platform'
    var_1 = 'linux'
    var_2 = 'test_script.py'
    var_3 = "print('success')"
    var_4 = 493
    var_5 = 'Popen'
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_pre_prompt_script. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt script exists.'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 9/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that work_in context manager is used (line 17 predicate evaluates to False).'
    var_1 = '/test/repo'
    var_2 = '/test/project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_6, var_2, var_5, var_7)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/11 statements.
# Partially parsed test_work_in_returns_to_original_directory. Retrieved 2/10 statements.
# Partially parsed test_work_in_with_none_stays_in_current_directory. Retrieved 1/7 statements.
# Partially parsed test_work_in_restores_directory_on_exception. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager changes to the specified directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in context manager returns to original directory after exit.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test that work_in restores original directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 14/27 statements.
# Partially parsed test_run_hook_from_repo_dir_restores_working_directory_on_exception. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = False

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
    var_13 = 'cookiecutter.hooks.logger'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
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

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir restores working directory even on exception.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = False



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts_found. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 2/9 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/11 statements.
# Partially parsed test_work_in_returns_to_original_directory_on_exception. Retrieved 4/15 statements.
# Partially parsed test_work_in_with_none_dirname. Retrieved 1/7 statements.
# Partially parsed test_work_in_with_path_object. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager changes to the specified directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even if exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test that work_in with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test that work_in works with Path objects.'
    var_1 = 'test_subdir'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist.'
    var_1 = 'test_repo'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_oserror_enoexec_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 12/23 statements.
# Partially parsed test_run_hook_from_repo_dir_returns_to_original_dir_on_exception. Retrieved 13/24 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = True

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't delete project when flag is False."
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = False

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
    var_8 = 'post_gen_project'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'Variable undefined'
    var_11 = module_0.UndefinedError(var_10)
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory during execution.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir returns to original directory even on exception.'
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
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_cleanup. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir successfully executes hook.'
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
    var_0 = 'Test run_hook_from_repo_dir cleans up project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on UndefinedError.'
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
    var_0 = "Test run_hook_from_repo_dir doesn't clean up when delete_project_on_failure is False."
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'post_gen_project'
    var_11 = False



# Parsed testcases at query #72
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
    var_0 = '/path/to/pre-commit'
    var_1 = 'post-commit'
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
    var_0 = '/different/path/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_find_hook_predicate_at_line_25_evaluates_to_false. Retrieved 17/33 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('hook')"
    var_3 = 'os.path.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'os.listdir'
    var_7 = [var_1]
    var_8 = lambda x: var_7
    var_9 = 'os.path.abspath'
    var_10 = lambda x: str(x)
    var_11 = 'os.path.join'
    var_12 = lambda x, y: f'{x}/{y}'
    var_13 = 'your_module.valid_hook'
    var_14 = 'pre_prompt'
    var_15 = module_0.find_hook(var_14, var_0)
    var_16 = len(var_15)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_script. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_with_bash_hook. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_directory. Retrieved 8/22 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_raises_exception. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid pre_prompt hook.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a bash hook script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt'
    var_4 = "#!/bin/bash\necho 'hook'"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when hook exists.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('executed')"
    var_5 = 493
    var_6 = 'test.txt'
    var_7 = 'content'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises exception when hook script fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'
    var_5 = 493



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 13/40 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'test_hook'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'hooks'
    var_8 = 'test_hook.sh'
    var_9 = 'test_hook'
    var_10 = 'hooks'
    var_11 = module_0.find_hook(var_9, var_10)
    var_12 = len(var_11)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/16 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 12/22 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 13/26 statements.
# Partially parsed test_run_hook_with_pathlib_path. Retrieved 10/19 statements.
# Partially parsed test_run_hook_with_string_project_dir. Retrieved 10/18 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'
    var_10 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook executes a single hook script.'
    var_1 = 'hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'
    var_11 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'hook1.sh'
    var_2 = 'hook2.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_gen_project'
    var_12 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook accepts Path objects.'
    var_1 = 'hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_prompt'

def test_case_0():
    var_0 = 'Test run_hook accepts string project directory.'
    var_1 = 'hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_prompt'

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns empty list.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'
    var_10 = 'No %s hook found'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/35 statements.
# Partially parsed test_run_script_with_context_with_python_script. Retrieved 12/33 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 10/24 statements.
# Partially parsed test_run_script_with_context_renders_complex_template. Retrieved 15/29 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.project_name }}'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'my_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'temp_script.sh'
    var_11 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script.'
    var_1 = "print('{{ cookiecutter.greeting }}')"
    var_2 = 'test_script.py'
    var_3 = 'cookiecutter'
    var_4 = 'greeting'
    var_5 = '_jinja2_env_vars'
    var_6 = 'Hello World'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'temp_script.py'
    var_11 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = "#!/usr/bin/env python\nprint('test')"
    var_2 = 'hook.py'
    var_3 = 'cookiecutter'
    var_4 = '_jinja2_env_vars'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.run_script_with_context(var_0, var_2, var_7)
    var_9 = 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja2 template.'
    var_1 = '{% for item in cookiecutter.items %}{{ item }}{% endfor %}'
    var_2 = 'template.sh'
    var_3 = 'cookiecutter'
    var_4 = 'items'
    var_5 = '_jinja2_env_vars'
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = {var_3: var_11}
    var_13 = module_0.run_script_with_context(var_0, var_2, var_12)
    var_14 = 0



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 12/23 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_preserves_exception. Retrieved 14/24 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Hook failed'
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_prompt'
    var_11 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Variable undefined'
    var_9 = module_0.UndefinedError(var_8)
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Hook failed'
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir re-raises the caught exception.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'Hook failed'
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = False
    var_11 = 'pre_prompt'
    var_12 = False
    var_13 = True
    assert var_13 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = 493
    var_3 = 'Popen'
    var_4 = 'MockProc'
    var_5 = ()
    var_6 = 'wait'
    var_7 = 0
    var_8 = lambda self: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/27 statements.
# Partially parsed test_run_script_with_context_with_jinja_variables. Retrieved 13/25 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 14/25 statements.
# Partially parsed test_run_script_with_context_empty_context. Retrieved 9/21 statements.
# Partially parsed test_run_script_with_context_uses_provided_cwd. Retrieved 10/23 statements.


def test_case_0():
    var_0 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_1 = 'test_script.sh'
    var_2 = 'utf-8'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'run_script'
    var_9 = 0
    var_10 = {var_8: var_9}
    var_11 = None
    var_12 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = '#!/bin/bash\necho "{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}"'
    var_1 = 'test_script.py'
    var_2 = 'utf-8'
    var_3 = 'cookiecutter'
    var_4 = 'var1'
    var_5 = 'var2'
    var_6 = 'hello'
    var_7 = 'world'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'echo "{{ cookiecutter.name }}"'
    var_1 = 'test_script.bat'
    var_2 = 'utf-8'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8[var_11]
    var_13 = '.bat'

def test_case_0():
    var_0 = '#!/bin/bash\necho "static content"'
    var_1 = 'test_script.sh'
    var_2 = 'utf-8'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = '#!/bin/bash\necho "test"'
    var_1 = 'test_script.sh'
    var_2 = 'utf-8'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'custom_dir'
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/20 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 10/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 12/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 10/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = len(var_6)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
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
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = len(var_6)
    assert var_11 == 0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.rmtree'
    var_8 = 'pre_prompt'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'post_gen_project'
    var_9 = False



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/15 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_prompt'
    var_9 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir keeps project on FailedHookException when flag is False.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_prompt'
    var_10 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'pre_prompt'
    var_11 = False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 18 (exit_status != EXIT_SUCCESS) evaluates to False.'
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'builtins.__import__'
    var_5 = 0
    var_6 = 0
    var_7 = var_5 != var_6
    assert var_7 is False



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 7/25 statements.
# Partially parsed test_run_script_windows_shell. Retrieved 6/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/20 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/21 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/21 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = None
    var_3 = False
    var_4 = 'Popen'
    var_5 = 'sys.platform'
    var_6 = 'linux'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = None
    var_3 = False
    var_4 = 'Popen'
    var_5 = 'sys.platform'
    var_6 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = None
    assert var_2 is True
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = None
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that exceptions are re-raised even when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project.py'
    var_8 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 14/28 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 14/29 statements.
# Partially parsed test_run_pre_prompt_hook_returns_path. Retrieved 14/29 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 14/33 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt scripts exist.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('pre_prompt hook')"
    var_5 = 'cookiecutter.hooks.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter.hooks.subprocess.Popen'
    var_9 = 'obj'
    var_10 = 'wait'
    var_11 = 0
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when pre_prompt script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('pre_prompt hook')"
    var_5 = 'cookiecutter.hooks.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter.hooks.subprocess.Popen'
    var_9 = 'obj'
    var_10 = 'wait'
    var_11 = 1
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a Path object.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 'cookiecutter.hooks.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter.hooks.subprocess.Popen'
    var_9 = 'obj'
    var_10 = 'wait'
    var_11 = 0
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"
    var_5 = 'cookiecutter.hooks.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter.hooks.subprocess.Popen'
    var_9 = 'obj'
    var_10 = 'wait'
    var_11 = 0
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 3/9 statements.
# Partially parsed test_run_pre_prompt_hook_creates_tmp_repo_and_runs_script. Retrieved 9/26 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_raises_exception. Retrieved 7/22 statements.
# Partially parsed test_run_pre_prompt_hook_string_repo_dir. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook creates a temp repo and runs the pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 493
    var_6 = []
    var_7 = 'Popen'
    var_8 = len(var_6)

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'
    var_5 = 493
    var_6 = 'Popen'

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook works with string repo_dir parameter.'
    var_1 = 'test_repo'
    var_2 = 'hooks'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/16 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 11/17 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 12/19 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook with a single script found.'
    var_1 = '/path/to/hook.py'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = [var_1]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook with multiple scripts found.'
    var_1 = '/path/to/hook1.py'
    var_2 = '/path/to/hook2.sh'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = [var_1, var_2]
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when scripts list is empty.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_prompt'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_find_hook_predicate_line_1. Retrieved 5/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = None
    var_4 = type(var_3)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('test')"
    var_3 = 'pre_prompt'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []
    var_4 = lambda hook_name: var_3



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test_hook'
    var_1 = ''



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 3/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Popen'
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = False
    var_6 = 'post_gen_project'



