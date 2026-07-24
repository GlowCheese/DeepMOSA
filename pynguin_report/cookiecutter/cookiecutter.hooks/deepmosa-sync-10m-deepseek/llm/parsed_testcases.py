####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py~'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown_hook.py'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/real_hook.py'
    var_1 = 'different_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/real_hook.py~'
    var_1 = 'different_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown.py~'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown_hook.py~'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_mismatched_name. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/14 statements.
# Partially parsed test_find_hook_without_extension. Retrieved 3/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'post_gen_project.py'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = ''



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 2/9 statements.
# Partially parsed test_run_hook_with_valid_script. Retrieved 5/17 statements.
# Partially parsed test_run_hook_with_jinja_script. Retrieved 9/21 statements.
# Partially parsed test_run_hook_ignores_backup_files. Retrieved 5/17 statements.
# Partially parsed test_run_hook_ignores_unsupported_hook_names. Retrieved 5/17 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 7/21 statements.
# Partially parsed test_run_hook_with_non_py_script. Retrieved 6/19 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hello")'
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("{{ cookiecutter.project_name }}")'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'TestProject'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = 'print("Backup")'
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = 'print("Unsupported")'
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("First")'
    var_3 = 'pre_gen_project.sh'
    var_4 = 'echo "Second"'
    var_5 = {}
    var_6 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.sh'
    var_2 = '#!/bin/bash\necho "Hello"'
    var_3 = 493
    var_4 = {}
    var_5 = 'pre_gen_project'



# Parsed testcases at query #4
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_hooks.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_empty_hook_file. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_hooks. Retrieved 6/17 statements.
# Partially parsed test_run_pre_prompt_hook_with_backup_file. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_unsupported_hook_name. Retrieved 3/10 statements.
# Failed to parse test_run_pre_prompt_hook_without_hooks_directory.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 0'
    var_5 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_no_hook_found. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_hooks_dir_is_not_a_directory. Retrieved 1/2 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/some/repo'
    var_1 = [var_0]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/10 statements.
# Partially parsed test_run_script_successful_shell_script. Retrieved 1/10 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 1/10 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 1/10 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = ''
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'import sys; sys.exit(0)'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_hooks.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook_extension. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_backup_file. Retrieved 5/15 statements.
# Partially parsed test_run_pre_prompt_hook_with_wrong_hook_name. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_empty_hook_file. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_hooks. Retrieved 6/17 statements.
# Failed to parse test_run_pre_prompt_hook_with_no_hooks_directory.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Pre-Prompt Hook script failed'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt.py~'
    var_4 = 'import sys\nsys.exit(1)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Hook script failed'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/bash\nexit 0'
    var_5 = 493



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/17 statements.
# Partially parsed test_run_hook_from_repo_dir_no_hook_found. Retrieved 12/13 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_failedhookexception. Retrieved 18/22 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_undefinederror. Retrieved 18/22 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_without_deletion. Retrieved 18/22 statements.


import cookiecutter.hooks as module_0

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = None
    var_13 = lambda hook_name, project_dir, context: var_12
    var_14 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = None
    var_10 = lambda hook_name: var_9
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = 'raise FailedHookException("Hook failed")'
    var_13 = exec(var_12)
    var_14 = lambda hook_name, project_dir, context: var_13
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = 'raise UndefinedError("Undefined variable")'
    var_13 = exec(var_12)
    var_14 = lambda hook_name, project_dir, context: var_13
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = 'raise FailedHookException("Hook failed")'
    var_13 = exec(var_12)
    var_14 = lambda hook_name, project_dir, context: var_13
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #13
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'some_hook'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None



# Parsed testcases at query #14
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_mismatched_name. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/14 statements.
# Partially parsed test_find_hook_without_extension. Retrieved 3/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'post_gen_project.py'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/21 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 4/7 statements.
# Partially parsed test_find_hook_with_multiple_valid_scripts. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'No hooks/dir in template_dir'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'post_gen_project.py'
    var_3 = [var_1, var_2]
    var_4 = ''
    var_5 = 'pre_gen_project'
    var_6 = 'pre_gen_project.py'
    var_7 = [var_1]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_no_hooks_dir. Retrieved 4/14 statements.
# Partially parsed test_run_hook_with_scripts. Retrieved 5/18 statements.
# Partially parsed test_run_hook_multiple_scripts. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project.py'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project.py'
    var_4 = 'pre_gen_project.sh'
    var_5 = 'pre_gen_project'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_no_hook_found_when_scripts_is_empty. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = 'project_name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/test'
    var_7 = [var_6]
    var_8 = 'pre_gen_project'
    var_9 = 'No pre_gen_project hook found'



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_hooks_dir. Retrieved 5/18 statements.
# Partially parsed test_find_hook_with_no_hooks_dir. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/10 statements.
# Partially parsed test_find_hook_with_no_matching_hook. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'any_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'any_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 7/9 statements.
# Partially parsed test_find_hook_with_invalid_hook_files. Retrieved 11/14 statements.
# Partially parsed test_find_hook_with_valid_hook_file. Retrieved 13/17 statements.
# Partially parsed test_find_hook_with_multiple_valid_hook_files. Retrieved 14/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = 'os.listdir'
    var_3 = []
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = 'os.listdir'
    var_3 = 'invalid.txt'
    var_4 = 'backup~'
    var_5 = [var_3, var_4]
    var_6 = 'os.path.abspath'
    var_7 = lambda x: f'/abs/{x}'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = module_0.find_hook(var_8, var_9)
    assert var_10 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = 'os.listdir'
    var_3 = 'pre_gen_project.py'
    var_4 = [var_3]
    var_5 = 'os.path.abspath'
    var_6 = lambda x: f'/abs/{x}'
    var_7 = 'os.path.join'
    var_8 = '/'
    var_9 = lambda *args: var_8.join(args)
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = bool(var_12 == ['/abs/hooks/pre_gen_project.py'])
    assert var_13 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = True
    var_2 = 'os.listdir'
    var_3 = 'pre_gen_project.py'
    var_4 = 'post_gen_project.py'
    var_5 = [var_3, var_4]
    var_6 = 'os.path.abspath'
    var_7 = lambda x: f'/abs/{x}'
    var_8 = 'os.path.join'
    var_9 = '/'
    var_10 = lambda *args: var_9.join(args)
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = module_0.find_hook(var_11, var_12)
    var_14 = bool(var_13 == ['/abs/hooks/pre_gen_project.py'])
    assert var_14 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 2/9 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/10 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'post_gen_project.py'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 1/4 statements.
# Partially parsed test_find_hook_with_no_matching_scripts. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = ''
    var_2 = 'pre_gen_project'



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'unknown'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/other_hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/other_hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py~'
    var_1 = 'unknown'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 6/42 statements.


def test_case_0():
    var_0 = 0
    var_1 = '/tmp/test_script.py'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'win'
    var_5 = 'No such file or directory'
    var_6 = 'Hook script failed (error:'
    var_7 = 'Exec format error'
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/11 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 2/13 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 1/11 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'
    var_1 = 493

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = ''
    var_1 = 420
    var_2 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'Hook script failed (error:'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_find_hook_no_matching_scripts. Retrieved 2/8 statements.
# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 1/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = ''
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project'



# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/18 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_invalid_hook_file. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_multiple_valid_files. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'post_gen_project.py'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_script_successful_python. Retrieved 1/10 statements.
# Partially parsed test_run_script_successful_non_python. Retrieved 1/10 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 1/10 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 1/10 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = ''
    var_1 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'import sys; sys.exit(0)'



# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_script_success_python. Retrieved 2/12 statements.
# Partially parsed test_run_script_success_non_python. Retrieved 3/14 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 2/12 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 3/14 statements.
# Partially parsed test_run_script_os_error_generic. Retrieved 2/8 statements.
# Partially parsed test_run_script_cwd. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'
    var_1 = '.'
    var_2 = [var_1]

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'
    var_1 = 493
    var_2 = '.'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = ''
    var_1 = 420
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'

def test_case_0():
    var_0 = '/non/existent/path/script.sh'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'import os; print(os.getcwd())'



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'unknown'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/other.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/.py'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.test.py'
    var_1 = 'hook.test'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #3
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_no_hooks_dir. Retrieved 2/9 statements.
# Partially parsed test_run_hook_empty_hooks_dir. Retrieved 3/12 statements.
# Partially parsed test_run_hook_no_matching_hook. Retrieved 5/16 statements.
# Partially parsed test_run_hook_with_backup_file. Retrieved 5/16 statements.
# Partially parsed test_run_hook_with_unsupported_hook_name. Retrieved 5/16 statements.
# Partially parsed test_run_hook_with_valid_hook. Retrieved 5/16 statements.
# Partially parsed test_run_hook_with_jinja_template. Retrieved 9/20 statements.
# Partially parsed test_run_hook_multiple_valid_hooks. Retrieved 7/20 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = {}
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.py'
    var_2 = ''
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = {}
    var_4 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hello from hook")'
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("{{ cookiecutter.project_name }}")'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'TestProject'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hook 1")'
    var_3 = 'pre_gen_project.sh'
    var_4 = 'echo "Hook 2"'
    var_5 = {}
    var_6 = 'pre_gen_project'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/18 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_mismatched_hook_name. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_no_hook_found_when_scripts_is_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/15 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_failure_with_deletion. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_failure_without_deletion. Retrieved 11/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 11/22 statements.


import cookiecutter.hooks as module_0

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
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = 'Undefined variable'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 5/13 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_mismatched_name. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/15 statements.
# Partially parsed test_find_hook_with_mixed_files. Retrieved 7/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = 0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project.sh'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project.py~'
    var_4 = 'unsupported.py'
    var_5 = 'pre_gen_project'
    var_6 = 0



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_empty_hooks_dir. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_valid_pre_prompt_script. Retrieved 3/12 statements.
# Partially parsed test_run_pre_prompt_hook_invalid_hook_file. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_backup_file. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_shell_script. Retrieved 4/14 statements.
# Partially parsed test_run_pre_prompt_hook_empty_script_file. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_scripts. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.txt'
    var_2 = 'test'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\nexit 0'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = ''
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt.sh'
    var_4 = '#!/bin/sh\nexit 0'
    var_5 = 493



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_hook_hooks_dir_not_directory. Retrieved 5/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = None
    var_2 = 'non_existent_dir'
    var_3 = 'some_hook'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #12
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'unknown'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/other.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/other.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py~'
    var_1 = 'unknown'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unknown.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_hooks_dir_is_not_directory. Retrieved 6/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter.hooks'
    var_1 = None
    var_2 = False
    var_3 = 'pre_gen_project'
    var_4 = 'non_existent_hooks'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 1/5 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_mismatching_hook_name. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/17 statements.
# Partially parsed test_find_hook_with_default_hooks_dir. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = ''
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = ''
    var_2 = 'unsupported_hook'

def test_case_0():
    var_0 = 'pre_gen_project.py~'
    var_1 = ''
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'post_gen_project.py'
    var_1 = ''
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = ''
    var_4 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3)



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_false. Retrieved 3/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'test_script.py'
    var_2 = module_0.run_script(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py~'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown_hook.py'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/other_hook.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown.py'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.py~'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/hook_name.txt'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/.py'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_no_hook_found_when_scripts_is_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_no_hook_found. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 5/18 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_invalid_hook_file_backup. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_invalid_hook_file_wrong_name. Retrieved 2/11 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 7/26 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = [var_3]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'
    var_6 = sorted(var_1)
    var_7 = bool(var_2 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_mismatched_name. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/14 statements.
# Partially parsed test_find_hook_without_extension. Retrieved 3/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'post_gen_project.py'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = ''



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_empty_hooks_dir. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_no_pre_prompt_script. Retrieved 4/12 statements.
# Partially parsed test_run_pre_prompt_hook_valid_pre_prompt_script. Retrieved 4/14 statements.
# Partially parsed test_run_pre_prompt_hook_valid_pre_prompt_script_with_shebang. Retrieved 4/14 statements.
# Partially parsed test_run_pre_prompt_hook_backup_file_ignored. Retrieved 4/12 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_empty_script_file. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_pre_prompt_scripts. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/bash\necho "test"'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("pre_prompt hook executed")'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = '#!/bin/bash\necho "pre_prompt hook executed"'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'
    var_2 = 'print("backup file")'
    var_3 = 493

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.sh'
    var_2 = ''
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("script1")'
    var_3 = 493
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "script2"'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_pre_prompt_script. Retrieved 9/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'pre_prompt.py'
    var_3 = [var_2]
    var_4 = 'pre_prompt'
    var_5 = None
    var_6 = lambda hook: var_3 if hook == var_4 else var_5
    var_7 = lambda script, repo: var_5
    var_8 = 'pre_prompt'
    var_9 = module_0.find_hook(var_8)
    var_10 = bool(var_9 is not None)
    assert var_10 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/14 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_failure_with_deletion. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_failure_without_deletion. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 11/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = 'Undefined variable'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_hook_returns_none_for_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_returns_none_for_no_matching_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_returns_none_for_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_returns_none_for_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_returns_list_for_single_valid_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_returns_list_for_multiple_valid_hooks. Retrieved 5/14 statements.
# Partially parsed test_find_hook_returns_list_for_valid_hook_without_extension. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'post_gen_project.py'
    var_4 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_pre_prompt_hook_without_pre_prompt_script. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'some_template'
    var_1 = [var_0]
    var_2 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/12 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 1/18 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 1/12 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 2/14 statements.
# Partially parsed test_run_script_cwd_parameter. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = ''
    var_1 = 420
    var_2 = 'might be an empty file or missing a shebang'
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'Hook script failed (error:'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'import sys; sys.exit(0)'



# Parsed testcases at query #31
#--------------------------






# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_pre_prompt_hook_without_pre_prompt_script. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/21 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_not_found. Retrieved 12/15 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_deletion. Retrieved 17/31 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_deletion. Retrieved 15/26 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 18/31 statements.


import cookiecutter.hooks as module_0

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name, hooks_dir='hooks': var_10
    var_12 = None
    var_13 = lambda script_path, cwd, context: var_12
    var_14 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = None
    var_10 = lambda hook_name, hooks_dir='hooks': var_9
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name, hooks_dir='hooks': var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name, hooks_dir='hooks': var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_16 = bool(False)
    assert var_16 is True

import jinja2.exceptions as module_0
import cookiecutter.hooks as module_1

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
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name, hooks_dir='hooks': var_10
    var_12 = ()
    var_13 = 'Undefined variable'
    var_14 = module_0.UndefinedError(var_13)
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_1.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #35
#--------------------------






# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------






# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 0
    var_1 = '/tmp/test_script.py'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'win'
    var_5 = 'Hook script failed, might be an empty file or missing a shebang'
    var_6 = False
    assert var_6 is False



# Parsed testcases at query #39
#--------------------------






