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
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.test.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/Pre-Commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_returns_original_repo_dir. Retrieved 3/8 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir_when_hook_exists. Retrieved 9/21 statements.
# Partially parsed test_run_pre_prompt_hook_with_python_hook. Retrieved 10/21 statements.
# Partially parsed test_run_pre_prompt_hook_failed_hook_raises_exception. Retrieved 7/19 statements.
# Partially parsed test_run_pre_prompt_hook_with_string_repo_dir. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_path_repo_dir. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'hooks'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temp directory and runs hook script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda script_path, cwd: var_7

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes Python hook script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"
    var_5 = []
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_5[var_8][var_8]

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when hook fails.'
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
    var_0 = 'Test run_pre_prompt_hook works with string repo_dir path.'
    var_1 = 'test_repo'
    var_2 = 'hooks'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook works with Path repo_dir.'
    var_1 = 'test_repo'
    var_2 = 'hooks'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_none_when_hook_is_backup_file. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_absolute_path_when_hook_exists. Retrieved 4/13 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 5/18 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt~'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = 'w'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook.sh'
    var_2 = 'w'
    var_3 = 'invalid_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []
    var_4 = lambda hook_name: var_3



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 8/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 8/22 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 6/24 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/25 statements.
# Partially parsed test_run_script_oserror. Retrieved 6/25 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'run_script.utils.make_executable'
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
    var_4 = 'run_script.utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1
    var_8 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'Popen'
    var_3 = 'run_script.utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = 'run_script.utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'run_script.utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'error:'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = "print('test')"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'run_script.utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = var_3[0][1]['cwd']



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/23 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 14/30 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 15/25 statements.
# Partially parsed test_run_script_with_context_uses_correct_cwd. Retrieved 14/24 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = 'test_script.py'
    var_2 = "print('Hello {{ cookiecutter.name }}')"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'World'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.message }}'"
    var_3 = 'cookiecutter'
    var_4 = 'message'
    var_5 = '_jinja2_env_vars'
    var_6 = 'Test message'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'tempfile.NamedTemporaryFile'
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_10)
    var_14 = bool(var_13 > 0)
    assert var_14 is True

def test_case_0():
    var_0 = 'Test run_script_with_context correctly renders Jinja2 templates.'
    var_1 = 'test_script.py'
    var_2 = 'name={{ cookiecutter.project_name }}\nauthor={{ cookiecutter.author }}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = '_jinja2_env_vars'
    var_7 = 'MyProject'
    var_8 = 'John Doe'
    var_9 = {}
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = 'name=MyProject'
    var_16 = bool('name=MyProject' in var_12[0])
    assert var_16 is True
    var_17 = 'author=John Doe'
    var_18 = bool('author=John Doe' in var_12[0])
    assert var_18 is True

def test_case_0():
    var_0 = 'Test run_script_with_context passes the correct working directory.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'cookiecutter'
    var_4 = '_jinja2_env_vars'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8[var_11]
    var_13 = str(var_12)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_true. Retrieved 7/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'sys.platform'
    var_1 = 'linux'
    var_2 = 'utils.make_executable'
    var_3 = 'subprocess.Popen'
    var_4 = 'Exec format error'
    var_5 = '/path/to/script.sh'
    var_6 = module_0.run_script(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'might be an empty file or missing a shebang'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_line_8_evaluates_to_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.py'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 3/11 statements.
# Partially parsed test_find_hook_returns_scripts_when_matching_hook_exists. Retrieved 4/15 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 6/21 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'content'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\necho test'
    var_4 = '#!/usr/bin/env python\nprint("test")'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'unsupported_hook'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 4/8 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 10/24 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 9/23 statements.
# Partially parsed test_run_pre_prompt_hook_early_return_with_scripts. Retrieved 6/16 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt script exists.'
    var_1 = 'template'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = None

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script successfully.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = None
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 'cookiecutter.hooks.create_tmp_repo_dir'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = None
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'Script failed'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns repo_dir early if scripts exist in original dir.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 'cookiecutter.hooks.find_hook'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt scripts.'
    var_1 = 'template'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt.py'
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = None
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 15/28 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 16/28 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 15/28 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without errors.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_prompt'
    var_14 = True
    var_15 = 'pre_prompt'

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Undefined'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'post_prompt'
    var_14 = True
    var_15 = 'post_prompt'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on exception when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_gen_project'
    var_14 = False
    var_15 = 'pre_gen_project'

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 8/10 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 11/22 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 15/30 statements.
# Partially parsed test_run_hook_passes_context_to_scripts. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = lambda hook_name: var_2
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook with a single script found.'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho "test"'
    var_3 = []
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'pre_prompt'
    var_10 = len(var_3)
    assert var_10 == 1
    var_11 = var_3[0][0]
    var_12 = var_3[0][1]
    var_13 = var_3[0][2]
    var_14 = bool(var_3[0][2] == var_8)
    assert var_14 is True

def test_case_0():
    var_0 = 'Test run_hook with multiple scripts found.'
    var_1 = 'post_gen_1.sh'
    var_2 = 'post_gen_2.sh'
    var_3 = '#!/bin/bash\necho "test1"'
    var_4 = '#!/bin/bash\necho "test2"'
    var_5 = []
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = 'cookiecutter.hooks.run_script_with_context'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'post_gen_project'
    var_14 = len(var_5)
    assert var_14 == 2
    var_15 = var_5[0][0]
    var_16 = var_5[1][0]
    var_17 = var_5[0][1]
    var_18 = var_5[1][1]
    var_19 = var_5[0][2]
    var_20 = bool(var_5[0][2] == var_12)
    assert var_20 is True
    var_21 = var_5[1][2]
    var_22 = bool(var_5[1][2] == var_12)
    assert var_22 is True

def test_case_0():
    var_0 = 'Test run_hook passes the context correctly to scripts.'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("test")'
    var_3 = []
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'
    var_12 = len(var_3)
    assert var_12 == 1
    var_13 = var_3[0]
    var_14 = bool(var_3[0] == var_10)
    assert var_14 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_script_path_ends_with_py. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = [var_1]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 6/18 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 6/19 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script. Retrieved 6/16 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_hooks. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid pre_prompt hook script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook executed')"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when hook exists.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises exception when hook script fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_5 = 493
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes all pre_prompt hooks.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hook1')"
    var_5 = 493
    var_6 = 'pre_prompt.sh'
    var_7 = "#!/bin/bash\necho 'hook2'"



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_script_path_ends_with_py. Retrieved 3/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that find_hook returns None when hooks directory does not exist.'
    var_1 = 'some_hook'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'pre_prompt'
    var_7 = False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'non_existent_hooks'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_hook_predicate_evaluates_to_false. Retrieved 6/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'nonexistent_hook'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 8/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'pre_prompt'
    var_6 = '.'
    var_7 = module_0.run_hook(var_5, var_6, var_4)
    var_8 = 'No pre_prompt hook found'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 7/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = 'Popen'
    var_2 = 'utils'
    var_3 = None
    var_4 = '/path/to/script'
    var_5 = '.'
    var_6 = module_0.run_script(var_4, var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 17/29 statements.
# Partially parsed test_run_script_with_context_with_jinja_vars. Retrieved 18/28 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 16/27 statements.
# Partially parsed test_run_script_with_context_renders_complex_template. Retrieved 15/26 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders script with context and executes it.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.name }}')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = 0
    var_15 = var_11[var_14]
    var_16 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context with custom jinja2 environment variables.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho '{{ variable }}'\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'variable'
    var_6 = '_jinja2_env_vars'
    var_7 = 'hello_world'
    var_8 = 'variable_start_string'
    var_9 = 'variable_end_string'
    var_10 = '[['
    var_11 = ']]'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = []
    var_16 = 'cookiecutter.hooks.run_script'
    var_17 = len(var_15)
    assert var_17 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = 'test_script.bat'
    var_2 = '@echo {{ cookiecutter.message }}\n'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'message'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_message'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = '.bat'

def test_case_0():
    var_0 = 'Test run_script_with_context renders complex jinja templates.'
    var_1 = 'complex_script.py'
    var_2 = "#!/usr/bin/env python\n# Project: {{ cookiecutter.project_name }}\n# Author: {{ cookiecutter.author }}\nprint('Setup complete')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = '_jinja2_env_vars'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = {}
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = []
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = 'my_project'
    var_16 = bool('my_project' in var_13[0])
    assert var_16 is True
    var_17 = 'John Doe'
    var_18 = bool('John Doe' in var_13[0])
    assert var_18 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = 'No %s hook found'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/5 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 3/7 statements.
# Partially parsed test_find_hook_single_matching_hook. Retrieved 6/13 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 8/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 7/16 statements.
# Partially parsed test_find_hook_no_matching_hook_name. Retrieved 5/11 statements.
# Partially parsed test_find_hook_unsupported_hook. Retrieved 5/11 statements.
# Partially parsed test_find_hook_absolute_path. Retrieved 5/16 statements.


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
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash'
    var_5 = 'pre_prompt'
    var_6 = module_0.find_hook(var_5, var_0)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt.py~'
    var_4 = 'pre_prompt'
    var_5 = module_0.find_hook(var_4, var_0)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'post_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'unsupported_hook'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python'
    var_3 = 'pre_prompt'
    var_4 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/20 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_cleanup. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = False
    var_10 = len(var_7)
    assert var_10 == 1
    var_11 = var_7[0][0]
    var_12 = bool(var_7[0][0] == var_6)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(not project_dir.exists())
    assert var_10 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(project_dir.exists())
    assert var_10 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(not project_dir.exists())
    assert var_10 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes from repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = False
    var_10 = var_7[var_9]
    var_11 = str(var_10)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_hook_with_no_scripts_found. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'find_hook'
    var_2 = 'run_script'
    var_3 = 0
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = 'cookiecutter.hooks.run_script_with_context'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_prompt'
    var_11 = '.'
    var_12 = [var_11]
    var_13 = var_4['find_hook']
    assert var_13 == 1
    var_14 = var_4['run_script']
    assert var_14 == 0



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used to change directory.'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory on hook failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 11/31 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test that tempfile.NamedTemporaryFile is called with delete=False, mode='wb', and correct suffix."
    var_1 = '/path/to/script.sh'
    var_2 = '/working/dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = None
    var_9 = module_0.run_script_with_context(var_1, var_2, var_7)
    var_10 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 15/26 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 15/23 statements.
# Partially parsed test_run_script_nonzero_exit_status. Retrieved 15/24 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 6/18 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/15 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 8/21 statements.


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
    var_10 = 'Popen'
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13

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
    var_10 = 'Popen'
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13

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
    var_11 = lambda *args, **kwargs: var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'workdir'
    var_2 = "print('success')"
    var_3 = {}
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = var_3['cwd']



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 3/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 5/16 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 5/16 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_no_hooks_dir. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no scripts exist.'
    var_1 = 'template'
    var_2 = 'hooks'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script successfully.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates temporary directory when scripts exist.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException on script failure.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys\nsys.exit(1)'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when hooks dir missing.'
    var_1 = 'template'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = None
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 4/12 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 6/17 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'non_existent_hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_other_file.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\necho test'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'post_gen_project'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true. Retrieved 4/29 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'Popen'
    var_2 = 'test_script.py'
    var_3 = 'win'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 12/22 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_preserves_project_on_success. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'post_gen_project'
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
    var_7 = 'undefined var'
    var_8 = module_0.UndefinedError(var_7)
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'post_gen_project'
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir does not delete project directory on success.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'post_gen_project'
    var_10 = True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_no_hooks_directory. Retrieved 3/8 statements.
# Partially parsed test_find_hook_empty_hooks_directory. Retrieved 4/11 statements.
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
    var_2 = 'print("hook")'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'print("backup")'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = 'print("hook")'
    var_3 = 'unsupported_hook'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("hook1")'
    var_3 = 'pre_prompt.sh'
    var_4 = 'echo "hook2"'
    var_5 = 'pre_prompt'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = len(var_7)
    assert var_9 == 2
    var_10 = bool(var_3 in var_7)
    assert var_10 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("hook")'
    var_3 = 'pre_prompt'
    var_4 = 'custom_hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 6/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_1)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    var_7 = bool(var_6 > 0)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 8/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 0
    var_4 = 'pre_prompt'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    var_7 = all(var_4)
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/24 statements.
# Partially parsed test_run_script_with_context_with_extensions. Retrieved 14/25 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 11/21 statements.
# Partially parsed test_run_script_with_context_complex_template. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test_value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1
    var_12 = var_9[0][1]

def test_case_0():
    var_0 = 'Test run_script_with_context with custom Jinja2 extensions.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.message }}'"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'message'
    var_6 = '_jinja2_env_vars'
    var_7 = 'hello world'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = 'test_script.bash'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.value }}'"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'value'
    var_6 = 'test123'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = var_9[0]
    assert var_11 == '.bash'

def test_case_0():
    var_0 = 'Test run_script_with_context with complex template expressions.'
    var_1 = 'test_script.py'
    var_2 = "#!/usr/bin/env python\nvar = '{{ cookiecutter.name|upper }}'"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'myproject'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = "var = 'MYPROJECT'"
    var_12 = bool("var = 'MYPROJECT'" in var_9[0])
    assert var_12 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_found. Retrieved 17/31 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# test hook'
    var_3 = 'os.path.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'os.listdir'
    var_7 = [var_1]
    var_8 = lambda x: var_7
    var_9 = 'os.path.join'
    var_10 = lambda x, y: f'{x}/{y}'
    var_11 = 'os.path.abspath'
    var_12 = lambda x: f'/abs/{x}'
    var_13 = 'valid_hook'
    var_14 = 'pre_prompt'
    var_15 = module_0.find_hook(var_14, var_0)
    var_16 = bool(var_15 is not None)
    assert var_16 is True
    var_17 = len(var_15)
    var_18 = bool(var_17 > 0)
    assert var_18 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('test')"
    var_3 = 'pre_prompt'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'utils.make_executable'
    var_5 = '/path/to/script.sh'
    var_6 = '.'
    var_7 = [var_6]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 2/8 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 4/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/12 statements.
# Partially parsed test_find_hook_returns_multiple_hook_scripts. Retrieved 6/17 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 3/8 statements.


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
    var_2 = 'print("hook")'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'print("backup")'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("hook1")'
    var_3 = 'pre_prompt.sh'
    var_4 = 'echo "hook2"'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("hook1")'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is called with repo_dir.'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 8/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'post_prompt'
    var_4 = 'post_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = 'post_prompt'
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = len(var_2)
    var_9 = bool(var_8 > 0)
    assert var_9 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'make_executable'
    var_5 = '/tmp/test_script.sh'
    var_6 = 0
    var_7 = var_6 != 0
    assert var_7 is False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used (line 17 predicate evaluates to False on exit).'
    var_1 = 'temp_repo'
    var_2 = 'temp_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'post_gen_project'
    var_8 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
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
    var_11 = 'pre_prompt'
    var_12 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
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
    var_0 = "Test run_hook_from_repo_dir doesn't delete when delete_project_on_failure is False."
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager_exits_normally. Retrieved 9/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 17 (work_in context manager) evaluates to False when exiting normally.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project.py'
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_0, var_6, var_2, var_5, var_7)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_exit_status_not_equal_to_exit_success. Retrieved 3/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed (exit status: 1)'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'might be an empty file or missing a shebang'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 9/21 statements.
# Partially parsed test_run_pre_prompt_hook_with_failed_hook. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_with_python_hook. Retrieved 8/19 statements.
# Partially parsed test_run_pre_prompt_hook_returns_path_object. Retrieved 9/21 statements.


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
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda script, cwd: var_7

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when hook script fails.'
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
    var_0 = 'Test run_pre_prompt_hook with a Python pre_prompt hook.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = None
    var_7 = lambda script, cwd: var_6

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a Path object when hook executes.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda script, cwd: var_7



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 9/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with the correct suffix from script_path.'
    var_1 = '/path/to/script.sh'
    var_2 = '/working/dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = module_0.run_script_with_context(var_1, var_2, var_5)
    var_8 = 1



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test that non-caught exceptions are raised when delete_project_on_failure is False.'
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
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 3/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = var_0.errno



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/32 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 16/27 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 15/24 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/17 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/16 statements.
# Partially parsed test_run_script_windows_shell. Retrieved 17/30 statements.


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
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    assert var_15 == 1
    var_16 = var_10[0]['cmd']
    var_17 = var_10[0]['cwd']

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
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    assert var_15 == 1
    var_16 = var_10[0]['cmd']

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
    var_10 = 'subprocess.Popen'
    var_11 = lambda cmd, shell=False, cwd='.': var_9
    var_12 = 'utils.make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subprocess.Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'shebang'

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
    var_1 = "print('test')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = 'sys.platform'
    var_12 = 'win32'
    var_13 = 'subprocess.Popen'
    var_14 = 'utils.make_executable'
    var_15 = None
    var_16 = lambda x: var_15
    var_17 = var_10[0]['shell']
    assert var_17 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/24 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 15/31 statements.
# Partially parsed test_run_script_with_context_with_jinja_filters. Retrieved 12/26 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 13/27 statements.
# Partially parsed test_run_script_with_context_different_cwd. Retrieved 14/29 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1
    var_12 = var_9[0][1]

def test_case_0():
    var_0 = 'Test that run_script_with_context properly renders Jinja2 templates.'
    var_1 = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test_value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'subprocess'
    var_11 = __import__(var_10)
    var_12 = var_11.Popen
    var_13 = 'subprocess.Popen'
    var_14 = len(var_9)
    var_15 = bool(var_14 > 0)
    assert var_15 is True
    var_16 = 'test_value'
    var_17 = bool('test_value' in var_9[0])
    assert var_17 is True

def test_case_0():
    var_0 = 'Test run_script_with_context with Jinja2 filters.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.text | upper }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'text'
    var_6 = 'hello'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'subprocess.Popen'
    var_11 = len(var_9)
    var_12 = bool(var_11 > 0)
    assert var_12 is True
    var_13 = 'HELLO'
    var_14 = bool('HELLO' in var_9[0])
    assert var_14 is True

def test_case_0():
    var_0 = 'Test that run_script_with_context preserves file extension.'
    var_1 = '#!/usr/bin/env python\nprint("test")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'subprocess.Popen'
    var_9 = len(var_7)
    var_10 = bool(var_9 > 0)
    assert var_10 is True
    var_11 = 0
    var_12 = var_7[var_11]
    var_13 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context respects different working directory.'
    var_1 = '#!/bin/bash\necho "test"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'different_dir'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = []
    var_9 = 'subprocess.Popen'
    var_10 = len(var_8)
    var_11 = bool(var_10 > 0)
    assert var_11 is True
    var_12 = 0
    var_13 = var_8[var_12]
    var_14 = str(var_13)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/20 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 7/18 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 7/20 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/17 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/16 statements.
# Partially parsed test_run_script_generic_oserror. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = len(var_1)
    assert var_6 == 1
    var_7 = var_1[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = len(var_1)
    assert var_6 == 1
    var_7 = var_1[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'subdir'
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = var_2[0][1]['cwd']

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'utils.make_executable'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exit status: 1'

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
    var_6 = 'error'
    var_7 = bool('error' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 3/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when pre_prompt hook scripts are not found.'
    var_1 = 'test_repo'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_work_in_context_manager_returns_to_original_directory. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager returns to original directory when exited.'
    var_1 = 'test_subdir'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'undefined variable'
    var_8 = 'cookiecutter.hooks.logger'
    var_9 = 'pre_prompt'
    var_10 = False

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory on hook failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'hook failed'
    var_8 = [var_7]
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'cookiecutter.hooks.logger'
    var_11 = 'pre_prompt'
    var_12 = True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir changes to repo_dir using work_in context manager.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'pre_prompt'
    var_5 = False
    var_6 = True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/24 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 12/24 statements.
# Partially parsed test_run_script_with_context_with_empty_context. Retrieved 10/21 statements.
# Partially parsed test_run_script_with_context_multiple_variables. Retrieved 14/25 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.project_name }}')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1
    var_12 = "print('my_project')"
    var_13 = bool("print('my_project')" in var_9[0])
    assert var_13 is True

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.message }}'\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'message'
    var_6 = 'Hello World'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1
    var_12 = var_9[0][1]
    assert var_12 == '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context works with empty cookiecutter context.'
    var_1 = 'test_script.py'
    var_2 = "print('no variables')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = "print('no variables')"
    var_11 = bool("print('no variables')" in var_7[0])
    assert var_11 is True

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple template variables.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.name }}'); print('{{ cookiecutter.version }}')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'version'
    var_7 = 'test_app'
    var_8 = '1.0.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = "print('test_app')"
    var_15 = bool("print('test_app')" in var_11[0])
    assert var_15 is True
    var_16 = "print('1.0.0')"
    var_17 = bool("print('1.0.0')" in var_11[0])
    assert var_17 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 8/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'subprocess.Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'utils.make_executable'
    var_5 = '/path/to/script.py'
    var_6 = '.'
    var_7 = module_0.run_script(var_5, var_6)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 14/28 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 15/28 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_directory. Retrieved 14/29 statements.


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
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

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
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_prompt'
    var_14 = True

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
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
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
    var_11 = 'pre_prompt'
    var_12 = False
    var_13 = str(var_9)



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 5/24 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 5/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 6/23 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/20 statements.
# Partially parsed test_run_script_oserror. Retrieved 6/19 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'Test running a Python script successfully.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'

def test_case_0():
    var_0 = 'Test running a shell script successfully.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho 'hello'"
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that returns non-zero exit status.'
    var_1 = 'test_script.py'
    var_2 = 'exit(1)'
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = module_0.run_script(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'exit status: 1'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that raises ENOEXEC error.'
    var_1 = 'test_script'
    var_2 = ''
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = module_0.run_script(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that raises OSError.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = module_0.run_script(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error'

def test_case_0():
    var_0 = 'Test running a script with custom working directory.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'cwd'
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_exit_status_not_equal_to_exit_success. Retrieved 3/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed (exit status: 1)'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_oserror_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test that project directory is not deleted when delete_project_on_failure is False.'
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
    var_12 = False



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that exceptions are re-raised when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'pre_prompt'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(project_dir.exists())
    assert var_10 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_run_pre_prompt_hook_predicate_false. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 evaluates to False when no pre_prompt hook exists.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.create_tmp_repo_dir'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 8/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'pre_prompt'
    var_5 = '.'
    var_6 = {}
    var_7 = module_0.run_hook(var_4, var_5, var_6)
    var_8 = 'No pre_prompt hook found'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = []
    var_4 = lambda hook_name: var_3



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_find_hook_no_hooks_directory. Retrieved 3/6 statements.
# Partially parsed test_find_hook_empty_directory. Retrieved 3/7 statements.
# Partially parsed test_find_hook_single_matching_hook. Retrieved 5/13 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 7/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 7/17 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 7/17 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/13 statements.


def test_case_0():
    var_0 = "Test find_hook when hooks directory doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook when hooks directory is empty.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with a single matching hook file.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '# hook content'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with multiple matching hook files with different extensions.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.sh'
    var_4 = '# python hook'
    var_5 = '#!/bin/bash'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores backup files ending with ~.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.py~'
    var_4 = '# hook content'
    var_5 = '# backup content'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = "Test find_hook ignores files that don't match the hook name."
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'post_gen_project.py'
    var_4 = '# matching hook'
    var_5 = '# non-matching hook'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns absolute paths.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '# hook content'
    var_4 = 'pre_prompt'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 7/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None
    var_2 = 'test_hook'
    var_3 = module_0.find_hook(var_2)
    assert var_3 is None
    var_4 = 'test_hook'
    var_5 = module_0.find_hook(var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 1/6 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 5/16 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_script. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_empty_hooks_dir. Retrieved 3/8 statements.
# Partially parsed test_run_pre_prompt_hook_no_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_string_path. Retrieved 2/7 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when no pre_prompt script exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script and returns new repo_dir.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns original repo_dir when hooks dir is empty.'
    var_1 = 'test_repo'
    var_2 = 'hooks'

def test_case_0():
    var_0 = "Test run_pre_prompt_hook returns original repo_dir when hooks dir doesn't exist."
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook works with string path input.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt scripts.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('script 1')"
    var_5 = 'pre_prompt.sh'
    var_6 = "#!/bin/bash\necho 'script 2'"



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/32 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 16/29 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 14/28 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 14/28 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 14/25 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
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
    var_12 = var_11()
    var_13 = lambda name, *args, **kwargs: __import__(name) if name != var_5 else var_12
    var_14 = '.'
    var_15 = len(var_2)
    assert var_15 == 1
    var_16 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'hello'"
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
    var_12 = var_11()
    var_13 = lambda name, *args, **kwargs: __import__(name) if name != var_5 else var_12
    var_14 = '.'
    var_15 = len(var_2)
    assert var_15 == 1
    var_16 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'Popen'
    var_3 = 'builtins.__import__'
    var_4 = 'utils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda name, *args, **kwargs: __import__(name) if name != var_4 else var_11
    var_13 = '.'
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid'
    var_2 = 'Popen'
    var_3 = 'builtins.__import__'
    var_4 = 'utils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda name, *args, **kwargs: __import__(name) if name != var_4 else var_11
    var_13 = '.'
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'test'
    var_2 = 'Popen'
    var_3 = 'builtins.__import__'
    var_4 = 'utils'
    var_5 = ()
    var_6 = 'make_executable'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda name, *args, **kwargs: __import__(name) if name != var_4 else var_11
    var_13 = '.'
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'Hook script failed (error:'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_prompt'
    var_11 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError.'
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
    var_0 = 'Test run_hook_from_repo_dir does not delete project on failure when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = False



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_run_script_with_context_temp_file_delete_false. Retrieved 9/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate delete=False at line 14 evaluates to False.'
    var_1 = "echo 'test'"
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/script.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 1



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup. Retrieved 14/29 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup. Retrieved 14/29 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_cleanup. Retrieved 14/29 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_without_cleanup. Retrieved 14/29 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = 'pre_prompt'
    var_11 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = None
    var_11 = 'Hook failed'
    var_12 = [var_11]
    var_13 = 'pre_prompt'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = None
    var_11 = 'Hook failed'
    var_12 = [var_11]
    var_13 = 'pre_prompt'
    var_14 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = None
    var_11 = 'Variable undefined'
    var_12 = 'pre_prompt'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up on UndefinedError when flag is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter.hooks.logger'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = None
    var_11 = 'Variable undefined'
    var_12 = 'pre_prompt'
    var_13 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_predicate_false. Retrieved 10/28 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used with repo_dir as dirname.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = 'post_gen_project'
    var_9 = False



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 12/24 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 11/21 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 13/24 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 10/21 statements.
# Partially parsed test_run_hook_from_repo_dir_restores_working_directory_on_exception. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully without errors.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'cookiecutter.hooks.rmtree'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False

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

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False.'
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

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Variable undefined'
    var_5 = module_0.UndefinedError(var_4)
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = []
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = var_3[0]

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir restores working directory even when exception occurs.'
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



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/33 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 15/30 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 13/27 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/22 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/21 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 15/30 statements.


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
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'utils.make_executable'
    var_13 = '.'
    var_14 = len(var_10)
    assert var_14 == 1
    var_15 = var_10[0]['cmd']
    var_16 = 'win'
    var_17 = var_10[0]['shell']

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
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'utils.make_executable'
    var_13 = '.'
    var_14 = len(var_10)
    assert var_14 == 1
    var_15 = var_10[0]['cmd']

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
    var_9 = var_8()
    var_10 = 'subprocess.Popen'
    var_11 = 'utils.make_executable'
    var_12 = '.'
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = '.'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = '.'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = '/custom/path'
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 0
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = var_9()
    var_11 = []
    var_12 = 'subprocess.Popen'
    var_13 = 'utils.make_executable'
    var_14 = len(var_11)
    assert var_14 == 1
    var_15 = var_11[0]['cwd']
    var_16 = bool(var_11[0]['cwd'] == var_2)
    assert var_16 is True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = None
    var_8 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_9 = 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 6/16 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_dir. Retrieved 5/15 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes pre_prompt script successfully.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary directory when scripts exist.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook raises FailedHookException when script fails.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys\nsys.exit(1)'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes multiple pre_prompt scripts.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('test1')"
    var_5 = 'pre_prompt.sh'
    var_6 = "#!/bin/bash\necho 'test2'"
    var_7 = 493



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 12/20 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 12/23 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 12/25 statements.
# Partially parsed test_run_hook_with_empty_scripts_list. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
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
    var_0 = 'Test run_hook when a single script is found.'
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
    var_0 = 'Test run_hook when multiple scripts are found.'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'post_gen_project'

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 8/25 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 8/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 6/21 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 6/18 statements.
# Partially parsed test_run_script_oserror. Retrieved 6/18 statements.
# Partially parsed test_run_script_default_cwd. Retrieved 8/22 statements.


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
    var_8 = var_2[0]['cmd']
    var_9 = var_2[0]['cwd']

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho test'
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = len(var_2)
    assert var_7 == 1
    var_8 = var_2[0]['cmd']

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Permission denied'

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
    var_8 = var_2[0]['cwd']
    assert var_8 == '.'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_script_path_ends_with_py. Retrieved 3/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/6 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 3/7 statements.
# Partially parsed test_find_hook_single_valid_hook. Retrieved 5/13 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 7/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/16 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 7/17 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/11 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/13 statements.


def test_case_0():
    var_0 = "Test find_hook returns None when hooks directory doesn't exist."
    var_1 = 'pre_prompt'
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = 'Test find_hook returns None when hooks directory is empty.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns list with single valid hook.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook returns list with multiple matching hooks.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/usr/bin/env python'
    var_5 = '#!/bin/bash'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores backup files ending with ~.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.py~'
    var_4 = '#!/usr/bin/env python'
    var_5 = 'pre_prompt'

def test_case_0():
    var_0 = "Test find_hook ignores hooks that don't match the hook_name."
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'post_gen_project.sh'
    var_4 = '#!/usr/bin/env python'
    var_5 = '#!/bin/bash'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores hooks that are not in supported hooks list.'
    var_1 = 'hooks'
    var_2 = 'unsupported_hook.py'
    var_3 = '#!/usr/bin/env python'
    var_4 = 'unsupported_hook'

def test_case_0():
    var_0 = 'Test find_hook returns absolute paths.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python'
    var_4 = 'pre_prompt'



# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

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
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
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
    var_0 = '/path/to/pre-commit.backup.sh'
    var_1 = 'pre-commit.backup'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid-hook~'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/usr/local/bin/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './hooks/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_predicate_satisfied. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'w'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 12/24 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 10/23 statements.
# Partially parsed test_run_pre_prompt_hook_returns_temp_dir. Retrieved 11/23 statements.


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
    var_8 = 'cookiecutter.hooks.utils.make_executable'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = len(var_6)
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 1'
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = 'cookiecutter.hooks.utils.make_executable'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns a temporary directory path.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = None
    var_8 = lambda x, cwd='.': var_7
    var_9 = 'cookiecutter.hooks.utils.make_executable'
    var_10 = lambda x: var_7
    var_11 = 'cookiecutter'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/8 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 4/11 statements.
# Partially parsed test_find_hook_single_matching_hook. Retrieved 7/18 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 9/24 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 9/23 statements.
# Partially parsed test_find_hook_no_matching_hook. Retrieved 6/16 statements.


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
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho "test"'
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\necho "test"'
    var_4 = '#!/usr/bin/env python\nprint("test")'
    var_5 = 'pre_prompt'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = len(var_7)
    assert var_9 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.sh~'
    var_3 = '#!/bin/bash\necho "test"'
    var_4 = '#!/bin/bash\necho "old"'
    var_5 = 'pre_prompt'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = var_7[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho "test"'
    var_3 = 'post_gen_project'
    var_4 = 'hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 3/6 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 3/7 statements.
# Partially parsed test_find_hook_single_matching_hook. Retrieved 5/13 statements.
# Partially parsed test_find_hook_multiple_matching_hooks. Retrieved 7/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 7/17 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/11 statements.
# Partially parsed test_find_hook_no_match_for_hook_name. Retrieved 5/11 statements.
# Partially parsed test_find_hook_mixed_files. Retrieved 11/26 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/13 statements.


def test_case_0():
    var_0 = "Test find_hook when hooks directory doesn't exist."
    var_1 = 'non_existent'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook when hooks directory is empty.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with a single matching hook file.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '# hook script'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook with multiple matching hook files (different extensions).'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.sh'
    var_4 = '# python hook'
    var_5 = '# shell hook'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test find_hook ignores backup files ending with ~.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.py~'
    var_4 = '# hook script'
    var_5 = '# backup'
    var_6 = 'pre_prompt'

def test_case_0():
    var_0 = "Test find_hook ignores files that don't match supported hooks."
    var_1 = 'hooks'
    var_2 = 'unsupported_hook.py'
    var_3 = '# unsupported'
    var_4 = 'unsupported_hook'

def test_case_0():
    var_0 = "Test find_hook when hook files exist but don't match the requested name."
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '# hook script'
    var_4 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test find_hook with a mix of matching, non-matching, and backup files.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'pre_prompt.sh'
    var_4 = 'pre_prompt.py~'
    var_5 = 'post_gen_project.py'
    var_6 = '# python'
    var_7 = '# shell'
    var_8 = '# backup'
    var_9 = '# other'
    var_10 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test that find_hook returns absolute paths.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '# hook script'
    var_4 = 'pre_prompt'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 8/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = 'pre_prompt'
    var_3 = '/tmp/project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.run_hook(var_2, var_3, var_6)
    var_8 = 'No pre_prompt hook found'



# Parsed testcases at query #12
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

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
    var_0 = '/path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/commit-msg'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-push.bak'
    var_1 = 'pre-push'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/usr/local/bin/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks/prepare-commit-msg'
    var_1 = 'prepare-commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_not_exists. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that find_hook returns None when hooks directory does not exist.'
    var_1 = 'non_existent_hooks'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_list_with_single_matching_hook. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/10 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks_with_different_extensions. Retrieved 6/19 statements.
# Partially parsed test_find_hook_uses_default_hooks_dir. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'non_existent'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho backup'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\necho test'
    var_4 = "#!/usr/bin/env python\nprint('test')"
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/24 statements.


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
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete when delete_project_on_failure is False.'
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



# Parsed testcases at query #16
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
    var_0 = '/path/to/commit-msg~'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/prepare-commit-msg.sh'
    var_1 = 'prepare-commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/prepare-commit-msg.sh~'
    var_1 = 'prepare-commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 7/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = module_0.find_hook(var_0, var_1)
    var_4 = f'{var_0}.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_hook_predicate_evaluates_to_false. Retrieved 8/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_other_hook.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = 0
    var_4 = str(var_1)
    var_5 = 'nonexistent_hook'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    assert var_7 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/14 statements.
# Partially parsed test_run_hook_executes_single_script. Retrieved 10/20 statements.
# Partially parsed test_run_hook_executes_multiple_scripts. Retrieved 13/26 statements.
# Partially parsed test_run_hook_passes_context_to_scripts. Retrieved 13/21 statements.
# Partially parsed test_run_hook_with_pathlib_path. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'pre_prompt'
    var_8 = 'No %s hook found'

def test_case_0():
    var_0 = 'Test run_hook executes a single hook script.'
    var_1 = 'pre_prompt.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_prompt'
    var_9 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'post_gen_project1.sh'
    var_2 = 'post_gen_project2.py'
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
    var_0 = 'Test run_hook passes context correctly to run_script_with_context.'
    var_1 = 'pre_prompt.sh'
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

def test_case_0():
    var_0 = 'Test run_hook works with pathlib.Path for project_dir.'
    var_1 = 'post_gen_project.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/14 statements.
# Partially parsed test_run_hook_with_single_script. Retrieved 12/22 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 13/26 statements.
# Partially parsed test_run_hook_passes_correct_hook_name. Retrieved 7/10 statements.
# Partially parsed test_run_hook_passes_correct_project_dir. Retrieved 10/19 statements.
# Partially parsed test_run_hook_passes_correct_context. Retrieved 13/21 statements.
# Partially parsed test_run_hook_with_path_object. Retrieved 10/19 statements.


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
    var_0 = 'Test run_hook when a single script is found.'
    var_1 = 'hook_script.sh'
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
    var_0 = 'Test run_hook when multiple scripts are found.'
    var_1 = 'hook_script_1.sh'
    var_2 = 'hook_script_2.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'pre_prompt'
    var_12 = 'Running hook %s'

def test_case_0():
    var_0 = 'Test run_hook passes the correct hook name to find_hook.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'prehooks_gen'

def test_case_0():
    var_0 = 'Test run_hook passes the correct project directory to run_script_with_context.'
    var_1 = 'hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook passes the correct context to run_script_with_context.'
    var_1 = 'hook.sh'
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
    var_12 = 'post_gen_project'

def test_case_0():
    var_0 = 'Test run_hook accepts Path object for project_dir.'
    var_1 = 'hook.sh'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'cookiecutter.hooks.logger'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'project'
    var_9 = 'pre_prompt'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda x: var_2
    var_4 = 'test_repo'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 16/25 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 17/24 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 8/18 statements.
# Partially parsed test_run_script_other_oserror. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'Test successful execution of a Python script.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 0
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = var_9()
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'Popen'
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14

def test_case_0():
    var_0 = 'Test successful execution of a non-Python script.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho hello'
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 0
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = var_9()
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'Popen'
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test execution with non-zero exit status raises FailedHookException.'
    var_1 = 'test_script.py'
    var_2 = 'exit(1)'
    var_3 = 'MockPopen'
    var_4 = ()
    var_5 = 'wait'
    var_6 = 1
    var_7 = lambda self: var_6
    var_8 = {var_5: var_7}
    var_9 = type(var_3, var_4, var_8)
    var_10 = var_9()
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'Popen'
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14
    var_16 = module_0.run_script(var_0)
    var_17 = bool(False)
    assert var_17 is True
    var_18 = 'exit status: 1'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test execution with ENOEXEC error raises FailedHookException.'
    var_1 = 'test_script'
    var_2 = ''
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = module_0.run_script(var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test execution with other OSError raises FailedHookException.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = module_0.run_script(var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'error:'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager_exits_gracefully. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 'Test that the work_in context manager at line 17 exits and restores the original directory.'
    var_1 = 'original'
    var_2 = 'repo'
    var_3 = 'project'
    var_4 = True
    var_5 = 'post_gen_project'
    var_6 = {}
    var_7 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 10/22 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 10/22 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory on hook failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'post_gen_project'
    var_9 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used correctly in run_pre_prompt_hook.'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 7/21 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 7/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/20 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/21 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/18 statements.


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
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed, might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
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
    var_11 = [var_10]
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete flag is False.'
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
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_prompt'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'Undefined variable'
    var_11 = module_0.UndefinedError(var_10)
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook from repo directory.'
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
    var_12 = len(var_9)
    assert var_12 == 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/12 statements.
# Partially parsed test_find_hook_returns_none_when_hooks_dir_is_empty. Retrieved 2/8 statements.
# Partially parsed test_find_hook_returns_matching_hook_file. Retrieved 5/16 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/12 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 4/12 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 6/17 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\nprint("test")'
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 15/38 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'test_hook'
    var_5 = 'hooks'
    var_6 = 'test_hook.sh'
    var_7 = '#!/bin/bash\necho "test"'
    var_8 = 'test_hook'
    var_9 = bool(var_1)
    assert var_9 is True
    var_10 = all(var_6)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = 'test_hook'
    var_13 = 'hooks'
    var_14 = module_0.find_hook(var_12, var_13)
    var_15 = all(var_10)
    var_16 = var_6 and var_15
    var_17 = bool(var_14 is None or var_16)
    assert var_17 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('hook')"
    var_3 = 'pre_prompt'
    var_4 = bool(var_1 > 0)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 12/38 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_prompt'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = 'pre_prompt.sh'
    var_7 = 0
    var_8 = None
    var_9 = len(var_2)
    assert var_9 == 1
    var_10 = 'non_existent_hook'
    var_11 = bool(var_2 is None or var_1)
    assert var_11 is True
    var_12 = all(var_10)
    var_13 = bool(var_12)
    assert var_13 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 14/30 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/33 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 14/34 statements.
# Partially parsed test_run_hook_from_repo_dir_no_cleanup_on_failure. Retrieved 14/34 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.work_in'
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'pre_prompt'
    var_12 = False
    var_13 = len(var_8)
    assert var_13 == 2
    var_14 = var_8[0][0]
    assert var_14 == 'work_in'
    var_15 = var_8[1][0]
    assert var_15 == 'run_hook'
    var_16 = var_8[1][1]
    assert var_16 == 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on FailedHookException when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.work_in'
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'pre_prompt'
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on UndefinedError when delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.work_in'
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = True
    var_14 = bool(False)
    assert var_14 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.work_in'
    var_10 = 'cookiecutter.hooks.run_hook'
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = False
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(var_8)
    assert var_15 == 0



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 9/21 statements.
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
    var_9 = bool(False)
    assert var_9 is True

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
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir deletes project directory on hook failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 5/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'test_hook'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/24 statements.
# Partially parsed test_run_script_with_context_python_extension. Retrieved 11/22 statements.
# Partially parsed test_run_script_with_context_with_jinja_variables. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_preserves_extension. Retrieved 12/23 statements.
# Partially parsed test_run_script_with_context_uses_correct_cwd. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.project_name }}'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 0
    var_9 = [var_8]
    var_10 = None
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = var_9[0]
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with .py script extension.'
    var_1 = "print('{{ cookiecutter.value }}')"
    var_2 = 'test_script.py'
    var_3 = 'cookiecutter'
    var_4 = 'value'
    var_5 = 'hello_world'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 0
    var_9 = [var_8]
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = var_9[0]
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context renders multiple Jinja variables.'
    var_1 = '{{ cookiecutter.name }}_{{ cookiecutter.version }}'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'version'
    var_6 = 'myapp'
    var_7 = '1.0.0'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = var_10[0]
    assert var_13 == 'myapp_1.0.0'

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension in temp file.'
    var_1 = "#!/usr/bin/env python\nprint('test')"
    var_2 = 'hook.py'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_6[var_9]
    var_11 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct working directory.'
    var_1 = 'echo test'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1
    var_9 = var_6[0]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_no_exception_when_delete_project_on_failure_false. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test that predicate at line 20 evaluates to False when no exception occurs.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 8/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to False when scripts are found.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = "#!/bin/bash\necho 'test'"
    var_4 = 'test_module'
    var_5 = None
    var_6 = lambda *args: var_5
    var_7 = 'os'
    var_8 = 0
    assert var_8 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_find_hook_returns_scripts_when_valid_hooks_exist. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '# hook script'
    var_3 = 'pre_prompt'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = 'post_gen_project'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = False



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_early_when_no_pre_prompt_script.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/11 statements.
# Partially parsed test_work_in_context_manager_returns_to_original_directory. Retrieved 2/12 statements.
# Partially parsed test_work_in_context_manager_with_none_dirname. Retrieved 1/7 statements.
# Partially parsed test_work_in_context_manager_predicate_evaluates_true. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager changes to the specified directory.'
    var_1 = 'test_subdir'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test that work_in context manager returns to original directory after exit.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in context manager with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test that the predicate at line 7 (with work_in(repo_dir):) evaluates to True.'
    var_1 = 'repo_dir'
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 9/26 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 9/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 7/21 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 7/22 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 7/21 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'Test running a Python script successfully.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = len(var_3)
    assert var_8 == 1
    var_9 = var_3[0]['cmd']
    var_10 = var_3[0]['cwd']

def test_case_0():
    var_0 = 'Test running a non-Python script successfully.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = len(var_3)
    assert var_8 == 1
    var_9 = var_3[0]['cmd']

def test_case_0():
    var_0 = 'Test running a script that returns non-zero exit status.'
    var_1 = 'test_script.py'
    var_2 = 'exit(1)'
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'Test running a script that raises OSError with ENOEXEC.'
    var_1 = 'test_script.py'
    var_2 = 'test'
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'Test running a script that raises OSError with other errno.'
    var_1 = 'test_script.py'
    var_2 = 'test'
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'Test that shell=True is used on Windows.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = []
    var_4 = 'Popen'
    var_5 = 'utils.make_executable'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'platform'
    var_9 = 'win32'
    var_10 = var_3[0]['shell']
    assert var_10 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 9/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the delete parameter in NamedTemporaryFile is False.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = "echo 'test'"
    var_5 = 'test.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_3)
    var_8 = 1



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_exit_status_equals_success. Retrieved 4/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_script.py'
    var_2 = '.'
    var_3 = module_0.run_script(var_1, var_2)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used (predicate at line 17 evaluates to False).'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'pre_prompt'
    var_5 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_no_deletion_when_delete_project_on_failure_false. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'Test that project directory is not deleted when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'cookiecutter.hooks.rmtree'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'post_gen_project'
    var_11 = False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_exit_status_not_equal_to_exit_success. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 0
    var_1 = '/path/to/script.py'
    var_2 = False
    var_3 = '/usr/bin/python3'
    var_4 = [var_3, var_1]
    var_5 = '.'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 8/25 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 6/19 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 3/16 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/16 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 3/13 statements.
# Partially parsed test_run_script_custom_cwd. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'builtins.__import__'
    var_5 = lambda name, *args: __import__(name)
    var_6 = '.'
    var_7 = len(var_2)
    assert var_7 == 1
    var_8 = var_2[0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'test'"
    var_2 = []
    var_3 = 'Popen'
    var_4 = '.'
    var_5 = len(var_2)
    assert var_5 == 1
    var_6 = var_2[0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'Popen'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = ''
    var_2 = 'Popen'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'test'
    var_2 = 'Popen'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Permission denied'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subdir'
    var_3 = []
    var_4 = 'Popen'
    var_5 = var_3[0][2]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 5/14 statements.
# Partially parsed test_run_pre_prompt_hook_returns_temp_dir. Retrieved 5/16 statements.
# Partially parsed test_run_pre_prompt_hook_string_path. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt scripts exist.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook executes a valid pre_prompt script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when script execution fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys; sys.exit(1)'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook returns a temporary directory path.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '# valid script'
    var_5 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook accepts string path.'
    var_1 = 'test_repo'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 5/11 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 6/14 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/11 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_same_name. Retrieved 8/20 statements.
# Partially parsed test_find_hook_with_custom_hooks_dir. Retrieved 6/14 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/13 statements.


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
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\necho test'
    var_4 = '#!/usr/bin/env python\nprint("test")'
    var_5 = 'pre_prompt'
    var_6 = module_0.find_hook(var_5, var_0)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'post_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existent_hooks'
    var_1 = 'test_hook'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 14/25 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 14/24 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 6/19 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 14/24 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/18 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/18 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 6/18 statements.


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
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'
    var_12 = 'sys.platform'
    var_13 = 'linux'

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
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'
    var_12 = 'sys.platform'
    var_13 = 'linux'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = {}
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = var_2['cwd']

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
    var_9 = var_8()
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'Popen'
    var_12 = 'sys.platform'
    var_13 = 'linux'
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid'
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
    var_6 = 'Permission denied'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = {}
    var_3 = 'Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = var_2['shell']
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts.




# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-push.sh'
    var_1 = 'pre-push'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid-hook'
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
    var_0 = '/path/to/pre-push.sh~'
    var_1 = 'pre-push'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/post-commit'
    var_1 = 'pre-commit'
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
    var_0 = './pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/17 statements.
# Partially parsed test_run_hook_with_scripts_found. Retrieved 12/23 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 12/24 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 11/17 statements.


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
    var_0 = 'Test run_hook when scripts are found and executed.'
    var_1 = 'hook_script.py'
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
    var_1 = 'hook_1.sh'
    var_2 = 'hook_2.py'
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter.hooks.logger'
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
    var_3 = 'cookiecutter.hooks.logger'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_prompt'
    var_10 = 'No %s hook found'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 11/22 statements.


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
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'post_gen_project'
    var_14 = False

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
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'post_gen_project'
    var_14 = True

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
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'pre_gen_project'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory during execution.'
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/26 statements.
# Partially parsed test_run_script_with_context_python_script. Retrieved 13/23 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 15/26 statements.
# Partially parsed test_run_script_with_context_calls_run_script_with_correct_cwd. Retrieved 10/20 statements.
# Partially parsed test_run_script_with_context_with_jinja2_env_vars. Retrieved 13/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.project_name }}'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'my_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.run_script_with_context(var_0, var_2, var_9)
    var_11 = 0
    var_12 = '.sh'
    var_13 = bool(var_6)
    assert var_13 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context with Python script.'
    var_1 = "print('{{ cookiecutter.message }}')"
    var_2 = 'test_script.py'
    var_3 = 'cookiecutter'
    var_4 = 'message'
    var_5 = '_jinja2_env_vars'
    var_6 = 'Hello World'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.run_script_with_context(var_0, var_2, var_9)
    var_11 = 0
    var_12 = '.py'
    var_13 = bool(var_6)
    assert var_13 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_script_with_context properly renders Jinja2 templates.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.name }}\necho {{ cookiecutter.version }}'
    var_2 = 'render_test.sh'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'version'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_project'
    var_8 = '1.0.0'
    var_9 = {}
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = module_0.run_script_with_context(var_0, var_2, var_11)
    var_13 = 0
    var_14 = 'utf-8'
    var_15 = 'test_project'
    var_16 = '1.0.0'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_script_with_context passes correct cwd to run_script.'
    var_1 = 'script.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'workdir'
    var_4 = 'cookiecutter'
    var_5 = '_jinja2_env_vars'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.run_script_with_context(var_0, var_1, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context respects _jinja2_env_vars.'
    var_1 = '{{ variable }}'
    var_2 = 'env_vars_test.sh'
    var_3 = 'cookiecutter'
    var_4 = 'variable'
    var_5 = '_jinja2_env_vars'
    var_6 = 'value'
    var_7 = 'trim_blocks'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = {var_3: var_10}
    var_12 = module_0.run_script_with_context(var_0, var_2, var_11)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/13 statements.
# Partially parsed test_run_hook_with_scripts_found. Retrieved 13/22 statements.
# Partially parsed test_run_hook_multiple_scripts. Retrieved 15/31 statements.
# Partially parsed test_run_hook_passes_correct_parameters. Retrieved 14/20 statements.
# Partially parsed test_run_hook_with_empty_scripts_list. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = None
    var_8 = lambda hook_name: var_7
    var_9 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook when scripts are found and executed.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_script.sh'
    var_7 = '#!/bin/bash\necho "test"'
    var_8 = None
    var_9 = lambda script, cwd, ctx: var_8
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test run_hook with multiple scripts found.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'script1.sh'
    var_7 = 'script2.sh'
    var_8 = '#!/bin/bash\necho "test1"'
    var_9 = '#!/bin/bash\necho "test2"'
    var_10 = []
    var_11 = 'cookiecutter.hooks.find_hook'
    var_12 = 'cookiecutter.hooks.run_script_with_context'
    var_13 = 'pre_prompt'
    var_14 = len(var_10)
    assert var_14 == 2

def test_case_0():
    var_0 = 'Test that run_hook passes correct parameters to run_script_with_context.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/path/to/script.py'
    var_7 = []
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = [var_6]
    var_10 = lambda hook_name: var_9
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'post_gen_project'
    var_13 = len(var_7)
    assert var_13 == 1
    var_14 = var_7[0][0]
    var_15 = bool(var_7[0][0] == var_6)
    assert var_15 is True
    var_16 = var_7[0][1]
    var_17 = var_7[0][2]
    var_18 = bool(var_7[0][2] == var_5)
    assert var_18 is True

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns empty list.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = []
    var_8 = lambda hook_name: var_7
    var_9 = 'pre_prompt'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'w'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_script_with_context_delete_false. Retrieved 9/22 statements.


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



# Parsed testcases at query #13
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_hook'
    var_1 = 'test_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/14 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_script_path_ends_with_py. Retrieved 4/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = '.py'



# Parsed testcases at query #16
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/pre-push'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/invalid-hook'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/commit-msg'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/prepare-commit-msg'
    var_1 = 'prepare-commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/15 statements.
# Partially parsed test_run_hook_scripts_found_and_executed. Retrieved 13/21 statements.
# Partially parsed test_run_hook_multiple_scripts_found. Retrieved 14/22 statements.
# Partially parsed test_run_hook_empty_scripts_list. Retrieved 10/15 statements.


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
    var_10 = 'No pre_prompt hook found'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when hook scripts are found and executed.'
    var_1 = '/hooks/pre_prompt.sh'
    var_2 = [var_1]
    var_3 = 'cookiecutter.hooks.find_hook'
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_prompt'
    var_11 = '/project'
    var_12 = module_0.run_hook(var_10, var_11, var_9)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when multiple hook scripts are found.'
    var_1 = '/hooks/post_gen_1.sh'
    var_2 = '/hooks/post_gen_2.py'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'post_gen_project'
    var_12 = '/project'
    var_13 = module_0.run_hook(var_11, var_12, var_10)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns an empty list.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda hook_name: var_2
    var_4 = 'pre_prompt'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.run_hook(var_4, var_5, var_8)
    var_10 = 'No pre_prompt hook found'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_oserror_enoexec_predicate_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = 'test_repo'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 3/9 statements.
# Partially parsed test_find_hook_returns_script_when_matching_hook_exists. Retrieved 5/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 3/9 statements.
# Partially parsed test_find_hook_returns_absolute_path. Retrieved 5/14 statements.
# Partially parsed test_find_hook_with_multiple_matching_scripts. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = '#!/usr/bin/env python\n'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = '#!/usr/bin/env python\n'
    var_2 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = '#!/usr/bin/env python\n'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_hook_no_hooks_dir. Retrieved 1/8 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 2/8 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 5/18 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/14 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/14 statements.
# Partially parsed test_find_hook_multiple_valid_hooks. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook.sh'
    var_2 = '#!/bin/bash\n'
    var_3 = 'invalid_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash\n'
    var_4 = '#!/usr/bin/env python\n'
    var_5 = 'pre_prompt'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts_returns_repo_dir. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'non_existent_hooks'
    var_1 = 'test_hook'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_directory. Retrieved 13/23 statements.


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
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = False

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
    var_10 = [var_9]
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'pre_prompt'
    var_13 = True

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
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
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
    var_11 = 'pre_prompt'
    var_12 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 9/28 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = 'nonexistent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'post_gen_project'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    var_7 = 'hooks'
    var_8 = 'post_gen_project'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 7/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'test_hook'
    var_4 = 'test_hook.sh'
    var_5 = 'test_hook'
    var_6 = bool(var_1)
    assert var_6 is True
    var_7 = len(var_2)
    var_8 = bool(var_7 > 0)
    assert var_8 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 8/19 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches FailedHookException at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'pre_prompt'
    var_7 = True

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir catches UndefinedError at line 20.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'pre_prompt'
    var_7 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 3/7 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 6/14 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 8/16 statements.
# Partially parsed test_find_hook_with_custom_hooks_dir. Retrieved 6/12 statements.


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
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/bin/bash'
    var_4 = '#!/usr/bin/env python'
    var_5 = 'pre_prompt'
    var_6 = module_0.find_hook(var_5, var_0)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\necho test'
    var_3 = 'post_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 2/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_find_hook_returns_list_of_strings_or_none. Retrieved 7/26 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None
    var_2 = 'test_hook'
    var_3 = module_0.find_hook(var_2)
    assert var_3 is None
    var_4 = 'test_hook'
    var_5 = module_0.find_hook(var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test that project directory is not deleted when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Test error'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_find_hook_returns_none_when_hooks_dir_does_not_exist. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_matching_hook_script. Retrieved 4/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 6/19 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/11 statements.
# Partially parsed test_find_hook_filters_unsupported_hooks. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unrelated_script.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt.sh'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = '#!/bin/bash\n'
    var_5 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'post_gen_project'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = 'unsupported_hook'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_script_path_when_hook_exists. Retrieved 4/13 statements.
# Partially parsed test_find_hook_returns_multiple_scripts_with_different_extensions. Retrieved 5/18 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/17 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = '/nonexistent/path'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.py'
    var_3 = 'w'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'pre_prompt.sh~'
    var_3 = 'w'
    var_4 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'
    var_4 = 0



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_absolute_path_for_matching_hook. Retrieved 5/17 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 5/16 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 6/16 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh~'
    var_2 = 'w'
    var_3 = 'pre_prompt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt.py'
    var_4 = 'pre_prompt'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = 'w'
    var_3 = 'pre_prompt'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 13/32 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test that NamedTemporaryFile is created with delete=False, mode='wb', and correct suffix."
    var_1 = '/path/to/script.sh'
    var_2 = '/working/dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = None
    var_9 = module_0.run_script_with_context(var_1, var_2, var_7)
    var_10 = False
    var_11 = 'wb'
    var_12 = '.sh'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 6/16 statements.
# Partially parsed test_find_hook_with_nonexistent_hooks_dir. Retrieved 3/4 statements.
# Partially parsed test_find_hook_with_no_matching_hooks. Retrieved 7/15 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 7/15 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 7/15 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 9/22 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'post_gen_project'
    var_5 = [var_1, var_4]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_other_hook'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = '__main__._HOOKS'
    var_4 = 'pre_prompt'
    var_5 = 'post_gen_project'
    var_6 = [var_4, var_5]

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
    var_1 = 'unsupported_hook'
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
    var_1 = '__main__._HOOKS'
    var_2 = 'pre_prompt'
    var_3 = 'post_gen_project'
    var_4 = [var_2, var_3]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_line_18_evaluates_to_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_find_hook_returns_scripts_list_when_valid_hooks_exist. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('hook')"
    var_3 = 'pre_prompt'
    var_4 = bool(var_1 > 0)
    assert var_4 is True
    var_5 = bool(var_2)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = []
    var_3 = lambda x: var_2
    var_4 = 'test_repo'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is called with repo_dir at line 7.'
    var_1 = 'test_repo_'
    var_2 = None



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')\n"
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_delete_false. Retrieved 13/31 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile.NamedTemporaryFile is called with delete=False.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_jinja2_env_vars'
    var_4 = 'test_project'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'echo "{{ cookiecutter.project_name }}"'
    var_9 = '/path/to/script.sh'
    var_10 = '/cwd'
    var_11 = module_0.run_script_with_context(var_9, var_10, var_7)
    var_12 = 1



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/27 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_delete. Retrieved 14/27 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 15/27 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
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
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_prompt'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'Hook failed'
    var_5 = [var_4]
    var_6 = 'cookiecutter.hooks.rmtree'
    var_7 = 'cookiecutter.hooks.logger'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_prompt'
    var_14 = False

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
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'pre_prompt'
    var_14 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
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
    var_13 = str(var_4)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 7/24 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 7/23 statements.
# Partially parsed test_run_script_windows_shell. Retrieved 6/22 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/21 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 5/22 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'subprocess.Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'success'"
    var_2 = []
    var_3 = 'subprocess.Popen'
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = var_2[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('success')"
    var_2 = []
    var_3 = 'subprocess.Popen'
    var_4 = 'sys.platform'
    var_5 = 'win32'
    var_6 = var_2[0][1]['shell']
    assert var_6 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'subprocess.Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subprocess.Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('test')"
    var_2 = 'subprocess.Popen'
    var_3 = 'sys.platform'
    var_4 = 'linux'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 9/21 statements.
# Partially parsed test_run_pre_prompt_hook_script_failure. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_copy. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'template'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt script.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.sh'
    var_4 = "#!/bin/bash\necho 'test'"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when the script fails.'
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
    var_0 = 'Test that run_pre_prompt_hook creates a temporary copy when hook exists.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "test"}'
    var_5 = 'pre_prompt.py'
    var_6 = "print('test')"
    var_7 = 493
    var_8 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_work_in_predicate_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 17 (dirname is not None) evaluates to False when dirname is None.'
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 14/31 statements.


import cookiecutter.hooks as module_0

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
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_7, var_2, var_5, var_8)
    var_10 = len(var_6)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_6[var_11]
    var_13 = str(var_12)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 9/30 statements.
# Partially parsed test_run_script_shell_file_success. Retrieved 5/21 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/23 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 3/19 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = []
    var_2 = []
    var_3 = 'Popen'
    var_4 = 'sys.executable'
    var_5 = '/usr/bin/python3'
    var_6 = 'make_executable'
    var_7 = len(var_1)
    assert var_7 == 1
    var_8 = var_1[0][0][0]
    var_9 = len(var_2)
    assert var_9 == 1

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = []
    var_2 = 'Popen'
    var_3 = 'make_executable'
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = var_1[0][0][0]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'Popen'
    var_2 = 'sys.executable'
    var_3 = '/usr/bin/python3'
    var_4 = 'make_executable'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'make_executable'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'might be an empty file or missing a shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'Popen'
    var_2 = 'make_executable'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed (error:'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 12/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that work_in context manager is used (line 17 predicate evaluates to False).'
    var_1 = '/tmp/test_repo'
    var_2 = '/tmp/test_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = False
    var_8 = 'pre_prompt'
    var_9 = False
    var_10 = module_0.run_hook_from_repo_dir(var_1, var_8, var_2, var_5, var_9)
    var_11 = 'pre_prompt'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_oserror_with_enoexec_errno. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 21 evaluates to True when OSError with ENOEXEC is raised.'
    var_1 = '/path/to/script.sh'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'Exec format error'
    var_5 = OSError(var_4)
    var_6 = 'might be an empty file or missing a shebang'



# Parsed testcases at query #51
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
    var_6 = 'error:'
    var_7 = 'shebang'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 10/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that tempfile is created with the correct suffix from script_path.'
    var_1 = "echo 'test'"
    var_2 = '/path/to/script.sh'
    var_3 = '/tmp'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = module_0.run_script_with_context(var_2, var_3, var_6)
    var_9 = len(var_7)
    var_10 = bool(var_9 > 0)
    assert var_10 is True
    var_11 = var_7[0][1]
    assert var_11 == '.sh'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_no_delete_on_failure. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir successfully runs a hook.'
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
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir during execution.'
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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_deletion. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_deletion. Retrieved 13/23 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 14/23 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/21 statements.


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
    var_10 = [var_9]
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
    var_0 = 'Test run_hook_from_repo_dir executes hook from repo directory.'
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



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 13/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_delete_on_failure. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_to_repo_dir. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook successfully.'
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
    var_13 = var_8[0][0]
    assert var_13 == 'post_gen_project'
    var_14 = var_8[0][1]
    var_15 = var_8[0][2]
    var_16 = bool(var_8[0][2] == var_7)
    assert var_16 is True

import cookiecutter.hooks as module_0

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
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_9, var_2, var_7, var_10)
    var_12 = bool(False)
    assert var_12 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project directory on failure.'
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
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_9, var_2, var_7, var_10)
    var_12 = bool(False)
    assert var_12 is True

import cookiecutter.hooks as module_0

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
    var_9 = 'post_gen_project'
    var_10 = True
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_9, var_2, var_7, var_10)
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes to repo directory during execution.'
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
    var_12 = var_8[0]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 10/24 statements.
# Partially parsed test_run_script_with_context_bash. Retrieved 10/22 statements.
# Partially parsed test_run_script_with_context_preserves_cwd. Retrieved 9/20 statements.
# Partially parsed test_run_script_with_context_with_env_vars. Retrieved 16/26 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.project_name }}')"
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = 'script_path'
    var_11 = bool('script_path' in var_8)
    assert var_11 is True
    var_12 = 'cwd'
    var_13 = bool('cwd' in var_8)
    assert var_13 is True
    var_14 = var_8['cwd']
    var_15 = var_8['content']
    assert var_15 == "print('test_project')"

def test_case_0():
    var_0 = 'Test run_script_with_context with bash script.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.name }}'"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'my_app'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = var_8[1]
    assert var_10 == "echo 'my_app'"

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct cwd to run_script.'
    var_1 = 'script.py'
    var_2 = "print('test')"
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'custom_dir'
    var_9 = var_6['value']

def test_case_0():
    var_0 = 'Test run_script_with_context respects _jinja2_env_vars.'
    var_1 = 'script.py'
    var_2 = '{{ variable }}'
    var_3 = 'cookiecutter'
    var_4 = 'variable'
    var_5 = '_jinja2_env_vars'
    var_6 = 'variable_start_string'
    var_7 = 'variable_end_string'
    var_8 = '[['
    var_9 = ']]'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 'test_value'
    var_13 = {var_3: var_11, var_4: var_12}
    var_14 = {}
    var_15 = 'cookiecutter.hooks.run_script'
    var_16 = var_14['value']
    assert var_16 == '{{ variable }}'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 13/26 statements.
# Partially parsed test_run_script_with_context_with_cookiecutter_vars. Retrieved 13/24 statements.
# Partially parsed test_run_script_with_context_preserves_file_extension. Retrieved 12/23 statements.
# Partially parsed test_run_script_with_context_renders_complex_template. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ greeting }} {{ name }}')\n"
    var_3 = 'greeting'
    var_4 = 'name'
    var_5 = 'cookiecutter'
    var_6 = 'Hello'
    var_7 = 'World'
    var_8 = {}
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = var_10[0][1]

def test_case_0():
    var_0 = 'Test run_script_with_context uses cookiecutter context variables.'
    var_1 = 'test_script.sh'
    var_2 = "echo '{{ cookiecutter.project_name }}'\n"
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = '_extensions'
    var_6 = 'my_project'
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves original file extension.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho test\n'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_6[var_9]
    var_11 = '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context handles complex Jinja2 templates.'
    var_1 = 'test_script.py'
    var_2 = "{% if enabled %}print('enabled'){% endif %}\nprint('{{ value }}')\n"
    var_3 = 'enabled'
    var_4 = 'value'
    var_5 = 'cookiecutter'
    var_6 = True
    var_7 = 'test_value'
    var_8 = {}
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = 'cookiecutter.hooks.run_script'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS.'
    var_1 = 'sys.platform'
    var_2 = 'linux'
    var_3 = 'sys.executable'
    var_4 = '/usr/bin/python3'
    var_5 = 'utils.make_executable'
    var_6 = 'subprocess.Popen'
    var_7 = '__main__.EXIT_SUCCESS'
    var_8 = 0
    var_9 = '/path/to/script.py'
    var_10 = '.'
    var_11 = [var_10]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/20 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 12/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 12/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 13/26 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/27 statements.


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
    var_9 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
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
    var_11 = 'pre_prompt'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
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
    var_11 = 'pre_prompt'
    var_12 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError and deletes project.'
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
    var_0 = 'Test run_hook_from_repo_dir changes to repo_dir before running hook.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = None
    var_4 = None
    var_5 = 'cookiecutter.hooks.run_hook'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'pre_prompt'
    var_10 = False
    var_11 = str(var_4)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 6/23 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 6/23 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 5/19 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/20 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = None
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(var_1 is not None)
    assert var_6 is True
    var_7 = var_1.args[0]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = None
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(var_1 is not None)
    assert var_6 is True
    var_7 = var_1.args[0]

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
    var_0 = 'test_script.py'
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
    var_6 = 'Hook script failed'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 23/45 statements.


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and executes script.'
    var_1 = 'test_script.py'
    var_2 = "#!/usr/bin/env python\n# Test script\nprint('{{ cookiecutter.name }}')\n"
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_jinja2_env_vars'
    var_6 = 'test_project'
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'subprocess.Popen'
    var_12 = 'make_executable'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = len(var_10)
    var_16 = bool(var_15 > 0)
    assert var_16 is True
    var_17 = 0
    var_18 = var_10[var_17][var_17]
    var_19 = var_18[var_17]
    var_20 = 'python.exe'
    var_21 = 1
    var_22 = var_18[var_21]
    var_23 = '.py'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_oserror_enoexec_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 22 evaluates to True when errno.ENOEXEC is raised.'
    var_1 = OSError()
    var_2 = var_1.errno



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 15/25 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_no_delete. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_changes_working_directory. Retrieved 12/23 statements.


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
    var_9 = 'Undefined'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'cookiecutter.hooks.logger'
    var_13 = 'post_gen_project'
    var_14 = True

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
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_hook'
    var_10 = 'pre_prompt'
    var_11 = False



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook_script. Retrieved 2/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook_script. Retrieved 9/20 statements.
# Partially parsed test_run_pre_prompt_hook_with_failed_hook_script. Retrieved 7/20 statements.
# Partially parsed test_run_pre_prompt_hook_creates_temp_repo. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when no pre_prompt hook exists.'
    var_1 = 'test_repo'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with a valid pre_prompt hook script.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook when hook script fails.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 493
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook creates a temporary repository.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "print('hook executed')"
    var_5 = 493
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    var_9 = bool(var_8 > 0)
    assert var_9 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 14/28 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 14/26 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 14/26 statements.
# Partially parsed test_run_script_oserror_enoexec. Retrieved 6/21 statements.
# Partially parsed test_run_script_oserror_other. Retrieved 6/21 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'hello'"
    var_2 = 'MockPopen'
    var_3 = ()
    var_4 = 'wait'
    var_5 = 0
    var_6 = lambda self: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = var_8()
    var_10 = 'Popen'
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12

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
    var_11 = 'utils.make_executable'
    var_12 = None
    var_13 = lambda x: var_12
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'exit status: 1'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'invalid script'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'shebang'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash'
    var_2 = 'Popen'
    var_3 = 'utils.make_executable'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = None
    assert var_2 == '/custom/path'
    var_3 = 'Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = '/custom/path'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_run_pre_prompt_hook_predicate_false. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 9 (if not scripts) evaluates to False.'
    var_1 = 'test_repo'
    var_2 = 'pre_prompt_script.sh'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = 0



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 9/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Popen'
    var_2 = 'sys.platform'
    var_3 = 'linux'
    var_4 = 'utils'
    var_5 = None
    var_6 = '/tmp/test_script.py'
    var_7 = '.'
    var_8 = module_0.run_script(var_6, var_7)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 11/26 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_delete. Retrieved 12/29 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_delete. Retrieved 12/29 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_no_delete. Retrieved 12/29 statements.


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes successfully.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on FailedHookException.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'Hook failed'
    var_7 = [var_6]
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir deletes project on UndefinedError.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'Undefined variable'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.work_in'
    var_4 = 'cookiecutter.hooks.run_hook'
    var_5 = 'cookiecutter.hooks.rmtree'
    var_6 = 'Hook failed'
    var_7 = [var_6]
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_run_pre_prompt_hook_work_in_context_manager. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'Test that work_in context manager is used at line 7.'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_work_in_context_manager. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir uses work_in context manager at line 17.'
    var_1 = 'post_gen_project'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_catches_failed_hook_exception. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_catches_undefined_error. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 9/20 statements.


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
    var_9 = bool(False)
    assert var_9 is True

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
    var_9 = bool(False)
    assert var_9 is True

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
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(not project_dir.exists())
    assert var_10 is True



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts_found. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'Test that predicate at line 9 (if not scripts) evaluates to True when no scripts found.'
    var_1 = 'test_repo'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_early_when_no_scripts_found. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter.hooks.find_hook'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_exit_status_not_equal_to_exit_success. Retrieved 3/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed (exit status: 1)'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 16/25 statements.
# Partially parsed test_run_script_shell_script_success. Retrieved 16/23 statements.
# Partially parsed test_run_script_non_zero_exit_status. Retrieved 17/24 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 8/18 statements.
# Partially parsed test_run_script_oserror. Retrieved 8/18 statements.
# Partially parsed test_run_script_calls_make_executable. Retrieved 15/25 statements.


def test_case_0():
    var_0 = 'Test running a Python script successfully.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'subprocess.Popen'
    var_4 = 'MockProc'
    var_5 = ()
    var_6 = 'wait'
    var_7 = 0
    var_8 = lambda self: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda *args, **kwargs: var_11
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14

def test_case_0():
    var_0 = 'Test running a shell script successfully.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho 'hello'"
    var_3 = 'subprocess.Popen'
    var_4 = 'MockProc'
    var_5 = ()
    var_6 = 'wait'
    var_7 = 0
    var_8 = lambda self: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda *args, **kwargs: var_11
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that fails with non-zero exit status.'
    var_1 = 'test_script.py'
    var_2 = 'exit(1)'
    var_3 = 'subprocess.Popen'
    var_4 = 'MockProc'
    var_5 = ()
    var_6 = 'wait'
    var_7 = 1
    var_8 = lambda self: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = lambda *args, **kwargs: var_11
    var_13 = 'utils.make_executable'
    var_14 = None
    var_15 = lambda x: var_14
    var_16 = module_0.run_script(var_0, var_1)
    var_17 = bool(False)
    assert var_17 is True
    var_18 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that raises ENOEXEC error.'
    var_1 = 'test_script'
    var_2 = ''
    var_3 = 'subprocess.Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = module_0.run_script(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test running a script that raises OSError.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = 'subprocess.Popen'
    var_4 = 'utils.make_executable'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = module_0.run_script(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'Test that run_script calls make_executable.'
    var_1 = 'test_script.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = 'subprocess.Popen'
    var_5 = 'MockProc'
    var_6 = ()
    var_7 = 'wait'
    var_8 = 0
    var_9 = lambda self: var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = var_11()
    var_13 = lambda *args, **kwargs: var_12
    var_14 = 'utils.make_executable'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_run_script_python_file_success. Retrieved 4/20 statements.
# Partially parsed test_run_script_non_python_file_success. Retrieved 4/18 statements.
# Partially parsed test_run_script_failure_non_zero_exit. Retrieved 5/19 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 5/19 statements.
# Partially parsed test_run_script_oserror. Retrieved 5/16 statements.
# Partially parsed test_run_script_uses_shell_on_windows. Retrieved 7/23 statements.
# Partially parsed test_run_script_default_cwd. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = "#!/bin/bash\necho 'hello'"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'exit(1)'
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = module_0.run_script(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script'
    var_1 = ''
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = module_0.run_script(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = 'subprocess.Popen'
    var_3 = 'utils.make_executable'
    var_4 = module_0.run_script(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = 'subprocess.Popen'
    var_4 = 'utils.make_executable'
    var_5 = 'sys.platform'
    var_6 = 'win32'
    var_7 = var_2[0]
    assert var_7 is True

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = 'subprocess.Popen'
    var_4 = 'utils.make_executable'
    var_5 = var_2[0]
    assert var_5 == '.'



