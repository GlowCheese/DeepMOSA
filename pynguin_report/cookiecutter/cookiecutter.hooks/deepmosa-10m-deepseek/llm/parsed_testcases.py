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
    var_0 = '/some/path/pre_commit.py'
    var_1 = 'other_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/pre_commit.py~'
    var_1 = 'other_hook'
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
    var_0 = '/some/path/unknown_hook.py~'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown.py~'
    var_1 = 'other_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_hook_no_hooks_found. Retrieved 2/9 statements.
# Partially parsed test_run_hook_single_hook_executed. Retrieved 7/19 statements.
# Partially parsed test_run_hook_multiple_hooks_executed. Retrieved 10/26 statements.
# Partially parsed test_run_hook_empty_context. Retrieved 5/17 statements.
# Partially parsed test_run_hook_with_none_project_dir. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {}

def test_case_0():
    var_0 = 'hook.py'
    var_1 = ''
    var_2 = 'pre_gen_project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}

def test_case_0():
    var_0 = 'hook1.py'
    var_1 = 'hook2.py'
    var_2 = ''
    var_3 = ''
    var_4 = 'pre_gen_project'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}
    var_9 = {var_5: var_6}

def test_case_0():
    var_0 = 'hook.py'
    var_1 = ''
    var_2 = 'post_gen_project'
    var_3 = {}
    var_4 = {}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook.py'
    var_1 = ''
    var_2 = 'pre_gen_project'
    var_3 = None
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.run_hook(var_2, var_3, var_6)
    var_8 = {var_4: var_5}



# Parsed testcases at query #3
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
    var_0 = '/some/path/other_hook.py~'
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
    var_0 = '/some/path/unknown_hook.py~'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path/unknown.py~'
    var_1 = 'hook_name'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 1/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_mismatched_hook_name. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 4/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project.py~'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = 'a'
    var_2 = 'unsupported_hook'

def test_case_0():
    var_0 = 'post_gen_project.py'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'pre_gen_project.sh'
    var_2 = 'a'
    var_3 = 'pre_gen_project'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_with_empty_hooks_dir. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook_file. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_backup_file. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_wrong_hook_name. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_python_script. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_shell_script. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_python_script. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_shell_script. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_empty_script_file. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_missing_shebang. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_valid_scripts. Retrieved 6/17 statements.
# Partially parsed test_run_pre_prompt_hook_creates_tmp_copy. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.txt'
    var_2 = 'test'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt~'
    var_2 = 'test'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'test'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\nexit 0'
    var_3 = 493
    var_4 = 'cookiecutter'

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
    var_2 = '#!/bin/sh\nexit 1'
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_1 = 'pre_prompt'
    var_2 = 'exit 0'
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\nexit 0'
    var_5 = 493
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'test.txt'
    var_4 = 'original'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_hooks_dir_is_not_a_directory. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = 'some_hook'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/17 statements.
# Partially parsed test_run_hook_from_repo_dir_no_hook_found. Retrieved 12/13 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_failed_hook_exception. Retrieved 17/25 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_undefined_error. Retrieved 18/25 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_without_deletion. Retrieved 17/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
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
    var_5 = 'Test'
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
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True

import jinja2.exceptions as module_0
import cookiecutter.hooks as module_1

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = ()
    var_13 = 'Undefined variable'
    var_14 = module_0.UndefinedError(var_13)
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_1.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = '/tmp/repo/hooks/pre_gen_project.py'
    var_10 = [var_9]
    var_11 = lambda hook_name: var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_delete_on_failure. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/project'
    var_2 = {}
    var_3 = 'pre_gen_project'
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_3, var_1, var_2, var_4)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = bool(var_1 == var_0)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_no_hooks_dir. Retrieved 2/11 statements.
# Partially parsed test_run_hook_with_scripts. Retrieved 3/15 statements.
# Partially parsed test_run_hook_multiple_scripts. Retrieved 4/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = {}
    var_1 = 'script.py'
    var_2 = 'post_gen_project'

def test_case_0():
    var_0 = {}
    var_1 = 'script1.py'
    var_2 = 'script2.py'
    var_3 = 'pre_gen_project'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_script_success_python. Retrieved 1/9 statements.
# Partially parsed test_run_script_success_non_python. Retrieved 1/9 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 1/9 statements.
# Partially parsed test_run_script_failure_enoexec. Retrieved 1/9 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 1/11 statements.


def test_case_0():
    var_0 = "print('hello')"

def test_case_0():
    var_0 = '#!/bin/sh\necho hello'

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'exit status: 1'

def test_case_0():
    var_0 = ''
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.sh'
    var_1 = module_0.run_script(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Hook script failed'

def test_case_0():
    var_0 = 'import os; print(os.getcwd())'



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 5/18 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 3/9 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 3/11 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 5/16 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 5/16 statements.
# Partially parsed test_find_hook_with_mismatched_hook_name. Retrieved 5/16 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 7/25 statements.
# Partially parsed test_find_hook_with_custom_hooks_dir. Retrieved 5/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'w'
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = 'w'
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'unsupported_hook.py'
    var_2 = 'w'
    var_3 = 'unsupported_hook'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'w'
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = 'w'
    var_4 = 'pre_gen_project'
    var_5 = module_0.find_hook(var_4, var_0)
    var_6 = sorted(var_5)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'custom_hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'w'
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 5/17 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/10 statements.
# Partially parsed test_find_hook_with_no_valid_hooks. Retrieved 2/10 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/23 statements.
# Partially parsed test_find_hook_uses_absolute_paths. Retrieved 5/18 statements.


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
    var_1 = 'pre_gen_project.py'
    var_2 = 'post_gen_project.py'
    var_3 = ''
    var_4 = 'pre_gen_project'
    var_5 = bool(var_1 == var_2)
    assert var_5 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = 0
    var_5 = bool(var_1)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_hook_no_hooks_found. Retrieved 2/9 statements.
# Partially parsed test_run_hook_with_valid_hook. Retrieved 5/16 statements.
# Partially parsed test_run_hook_with_jinja_context. Retrieved 9/20 statements.
# Partially parsed test_run_hook_multiple_scripts. Retrieved 8/22 statements.
# Partially parsed test_run_hook_invalid_hook_ignored. Retrieved 7/20 statements.
# Partially parsed test_run_hook_backup_file_ignored. Retrieved 5/16 statements.


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
    var_2 = '{{ cookiecutter.project_name }}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'TestProject'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("First")'
    var_3 = 'pre_gen_project.sh'
    var_4 = '#!/bin/bash\necho "Second"'
    var_5 = 493
    var_6 = {}
    var_7 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Valid")'
    var_3 = 'invalid_hook.py'
    var_4 = 'print("Invalid")'
    var_5 = {}
    var_6 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = 'print("Backup")'
    var_3 = {}
    var_4 = 'pre_gen_project'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_no_hook_found. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'pre_gen_project'
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_with_empty_hooks_dir. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_with_non_matching_hook. Retrieved 2/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_backup_file. Retrieved 2/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_python_hook. Retrieved 3/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_shell_hook. Retrieved 4/12 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_python_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_shell_hook. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_empty_hook_file. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_missing_shebang. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_multiple_valid_hooks. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py~'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\nexit 0'
    var_3 = 493

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
    var_2 = '#!/bin/sh\nexit 1'
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_1 = 'pre_prompt'
    var_2 = 'exit 0'
    var_3 = 493
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\nexit 0'
    var_5 = 493



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/18 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/17 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 4/17 statements.
# Partially parsed test_find_hook_with_mismatched_hook_name. Retrieved 4/17 statements.
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
    var_2 = 'pre_gen_project.txt'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/18 statements.
# Partially parsed test_find_hook_with_no_hooks_dir. Retrieved 2/8 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/10 statements.
# Partially parsed test_find_hook_with_no_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.sh'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'any_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'any_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'some_file.txt'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.sh'
    var_2 = 'post_gen_project.sh'
    var_3 = [var_1, var_2]
    var_4 = ''
    var_5 = 'any_hook'



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/18 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/9 statements.
# Partially parsed test_find_hook_with_no_matching_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 6/21 statements.


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
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = [var_1, var_2]
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/5 statements.
# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/11 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_mismatched_name. Retrieved 4/9 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/14 statements.


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
    var_1 = 'unsupported_hook.py'
    var_2 = ''
    var_3 = 'unsupported_hook'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py~'
    var_2 = ''
    var_3 = 'pre_gen_project'

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_on_failure_when_flag_false. Retrieved 6/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_empty_hooks_dir. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_invalid_hook_files. Retrieved 4/15 statements.
# Partially parsed test_run_pre_prompt_hook_valid_python_script. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_valid_shell_script. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_script_fails. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_empty_script_file. Retrieved 3/12 statements.
# Partially parsed test_run_pre_prompt_hook_multiple_valid_scripts. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'hooks'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.txt'
    var_2 = 'pre_prompt~'
    var_3 = 'post_gen_project.py'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\nexit 0'
    var_3 = 493
    var_4 = 'cookiecutter'

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
    var_2 = 493
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\nexit 0'
    var_5 = 493
    var_6 = 'cookiecutter'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_hook_no_hooks_found. Retrieved 2/9 statements.
# Partially parsed test_run_hook_with_valid_script. Retrieved 5/17 statements.
# Partially parsed test_run_hook_with_multiple_valid_scripts. Retrieved 8/23 statements.
# Partially parsed test_run_hook_ignores_backup_files. Retrieved 5/17 statements.
# Partially parsed test_run_hook_ignores_unsupported_hook_names. Retrieved 5/17 statements.
# Partially parsed test_run_hook_with_jinja_context. Retrieved 7/19 statements.


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
    var_2 = 'print("Hello")'
    var_3 = 'pre_gen_project.sh'
    var_4 = '#!/bin/bash\necho "World"'
    var_5 = 493
    var_6 = {}
    var_7 = 'pre_gen_project'

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
    var_2 = 'print("{{ greeting }}")'
    var_3 = 'greeting'
    var_4 = 'Hello World'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_hook_no_hooks_dir. Retrieved 2/9 statements.
# Partially parsed test_run_hook_empty_hooks_dir. Retrieved 3/13 statements.
# Partially parsed test_run_hook_no_matching_script. Retrieved 5/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = {}
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = {}
    var_4 = 'pre_gen_project'

def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_mismatched_hook_name. Retrieved 3/12 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project.py~'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = 'a'
    var_2 = 'unsupported_hook'

def test_case_0():
    var_0 = 'post_gen_project.py'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'pre_gen_project.sh'
    var_2 = 'a'
    var_3 = 'pre_gen_project'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_hooks_dir_is_not_directory. Retrieved 1/2 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/17 statements.
# Partially parsed test_run_hook_from_repo_dir_no_hook_found. Retrieved 12/13 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_failed_hook_exception. Retrieved 17/25 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_undefined_error. Retrieved 18/25 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_without_deletion. Retrieved 17/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
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
    var_4 = 'name'
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
    var_4 = 'name'
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

import jinja2.exceptions as module_0
import cookiecutter.hooks as module_1

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'name'
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
    var_15 = None
    var_16 = lambda path: var_15
    var_17 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/some/repo'
    var_1 = [var_0]



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 3/16 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'a'
    var_2 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'post_gen_project.py'
    var_2 = 'a'
    var_3 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 1/4 statements.
# Partially parsed test_find_hook_no_matching_script. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'other_hook.py'
    var_2 = 'w'
    var_3 = 'pre_gen_project'



# Parsed testcases at query #14
#--------------------------






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

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = len(var_2)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_returns_list_when_valid_hook_exists. Retrieved 5/20 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 5/18 statements.
# Partially parsed test_find_hook_filters_by_valid_hook. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'
    var_4 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = ''
    var_3 = 'post_gen_project'
    var_4 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = 'pre_gen_project'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 'pre_gen_project.py'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_no_hook_found_when_scripts_is_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = 'non_existent_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 15/19 statements.
# Partially parsed test_run_hook_from_repo_dir_no_hook. Retrieved 12/15 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_with_deletion. Retrieved 16/26 statements.
# Partially parsed test_run_hook_from_repo_dir_hook_fails_without_deletion. Retrieved 15/24 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 17/26 statements.


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
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda script, cwd, ctx: var_12
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
    var_10 = lambda x: var_9
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
    var_11 = lambda x: var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = None
    var_16 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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
    var_11 = lambda x: var_10
    var_12 = ()
    var_13 = 'Hook failed'
    var_14 = [var_13]
    var_15 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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
    var_11 = lambda x: var_10
    var_12 = ()
    var_13 = 'Undefined variable'
    var_14 = module_0.UndefinedError(var_13)
    var_15 = None
    var_16 = module_1.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/10 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 1/10 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 1/10 statements.
# Partially parsed test_run_script_enoexec_error. Retrieved 2/12 statements.
# Partially parsed test_run_script_with_cwd. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'import sys; sys.exit(0)'

def test_case_0():
    var_0 = '#!/bin/sh\nexit 0'

def test_case_0():
    var_0 = 'import sys; sys.exit(1)'
    var_1 = 'Hook script failed (exit status: 1)'

def test_case_0():
    var_0 = ''
    var_1 = 292
    var_2 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = 'Hook script failed (error:'

def test_case_0():
    var_0 = 'import sys; sys.exit(0)'
    var_1 = '/tmp'
    var_2 = [var_1]



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_hook_empty_hooks_dir. Retrieved 1/4 statements.
# Partially parsed test_find_hook_no_matching_scripts. Retrieved 3/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'pre_gen_project'

def test_case_0():
    var_0 = 'other_hook.py'
    var_1 = ''
    var_2 = 'pre_gen_project'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/12 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_with_deletion. Retrieved 10/16 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception_without_deletion. Retrieved 10/16 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error_with_deletion. Retrieved 10/16 statements.
# Partially parsed test_run_hook_from_repo_dir_work_in_context. Retrieved 10/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
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
    var_5 = 'Test'
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
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
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
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = {}
    var_3 = '.'
    var_4 = [var_3]
    var_5 = 'pre_gen_project'
    var_6 = 'No pre_gen_project hook found'



# Parsed testcases at query #26
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_oserror_errno_not_enoexec. Retrieved 2/5 statements.


def test_case_0():
    var_0 = OSError()
    var_1 = var_0.errno



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/12 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 1/18 statements.
# Partially parsed test_run_script_fails_with_non_zero_exit. Retrieved 1/12 statements.
# Partially parsed test_run_script_fails_with_enoexec. Retrieved 1/12 statements.
# Partially parsed test_run_script_fails_with_oserror. Retrieved 2/12 statements.


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

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'import sys; sys.exit(0)'
    var_2 = 'Hook script failed (error:'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_in_directory. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_no_hooks_directory. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_empty_hooks_directory. Retrieved 2/8 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_unsupported_hook_name. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_matching_hook_but_different_extension. Retrieved 4/15 statements.
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
    var_1 = 'pre_gen_project.sh'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'post_gen_project.py'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #30
#--------------------------






