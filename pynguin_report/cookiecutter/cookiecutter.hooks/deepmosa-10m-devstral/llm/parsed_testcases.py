####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'post-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown-hook.py'
    var_1 = 'unknown-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.txt'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'tests/test-templates/pre-post-hooks/'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/test-templates/pre-post-hooks/'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/test-templates/pre-post-hooks/'
    var_1 = [var_0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/post-commit'



# Parsed testcases at query #4
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = '/fake/dir'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/dir/hooks/pre_gen_project.py'
    var_8 = [var_7]
    var_9 = module_0.find_hook(var_0)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True
    var_11 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False
    var_3 = 'pre-commit.py'
    var_4 = module_0.valid_hook(var_3, var_1)
    assert var_4 is True
    var_5 = 'invalid-hook.py'
    var_6 = module_0.valid_hook(var_5, var_1)
    assert var_6 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.py'
    var_6 = [var_5]
    var_7 = 'print("Hello, {{ cookiecutter.project_name }}!")'
    var_8 = 'utf-8'
    var_9 = '.'
    var_10 = [var_9]



# Parsed testcases at query #7
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 10/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_0]
    var_8 = '#!/bin/bash\necho "test"'
    var_9 = 'utf-8'
    var_10 = module_0.run_script_with_context(var_0, var_1, var_6)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #10
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'test_hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'test_hooks_dir/pre-commit'
    var_8 = 'test_hooks_dir/pre-commit.another'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = False



# Parsed testcases at query #15
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 2/3 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = [var_0]



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_hook(var_0, var_1, var_6)
    assert var_7 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/path/to/hook/script.py'
    var_8 = [var_7]
    var_9 = module_0.run_hook(var_0, var_1, var_6)
    assert var_9 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = 'test'
    var_4 = 'context'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/some/repo'
    var_1 = [var_0]



# Parsed testcases at query #21
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'post-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown-hook.py'
    var_1 = 'unknown-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.txt'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = True
    var_2 = bool(var_1 == var_0)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = '/test/path'
    var_2 = 'test'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_hook(var_0, var_1, var_6)
    var_8 = 'No %s hook found'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/9 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/8 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/9 statements.
# Partially parsed test_run_script_os_error_empty_file. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = False
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = [var_0]
    var_3 = False
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = False
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'empty file or missing a shebang'
    var_5 = bool('empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'error:'
    var_5 = bool('error:' in var_3)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/post-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.another'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_hook_failure_without_deletion. Retrieved 11/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_oserror_with_enonexec_raises_failed_hook_exception. Retrieved 3/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'No exec'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/data/cookiecutter-no-hooks'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/cookiecutter-with-hooks'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/cookiecutter-failing-hook'
    var_1 = [var_0]



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 6/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/test/directory'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 8/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = 'test_key'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}
    var_5 = [var_0]
    var_6 = 'test content'
    var_7 = 'utf-8'
    var_8 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_9 = [var_0]



# Parsed testcases at query #34
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = False
    var_8 = 'wb'
    var_9 = '.sh'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_returns_list_of_absolute_paths_for_valid_hooks. Retrieved 3/8 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent-hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = '~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported-hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #37
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #38
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = 'pre_gen_project'
    var_2 = '/some/project/dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook(var_1, var_2, var_7)
    var_10 = bool(not var_8)
    assert var_10 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_hook_failure_without_deletion. Retrieved 9/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'fake'
    var_4 = 'context'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = 'Hook failed'
    var_8 = [var_7]
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_scripts.




# Parsed testcases at query #41
#--------------------------

# Failed to parse test_work_in_context_manager_is_used.




# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate. Retrieved 10/15 statements.


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = [var_0]
    var_2 = 'pre_gen_project'
    var_3 = '/fake/project'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True
    var_11 = 'Hook failed'
    var_12 = [var_11]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 6/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = 'test_hook'
    var_2 = 'test_project_dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_oserror_with_enexec_raises_failed_hook_exception. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Exec format error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_oserror_handling. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed, might be an empty file or missing a shebang'



# Parsed testcases at query #48
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #49
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(str(e).startswith('Hook script failed (error:'))
    assert var_3 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_when_delete_project_on_failure_is_false. Retrieved 9/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = [var_0]
    var_2 = '/fake/repo'
    var_3 = 'pre_gen_project'
    var_4 = '/fake/project'
    var_5 = 'test'
    var_6 = 'context'
    var_7 = {var_5: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_2, var_3, var_4, var_7, var_8)



# Parsed testcases at query #51
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'failing_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'failing_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'undefined_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'undefined_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hooks. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/test-templates/pre-prompt-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/test-templates/pre-prompt-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/test-templates/pre-prompt-hook-fail'
    var_1 = [var_0]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 9/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_6 = False
    var_7 = 'wb'
    var_8 = '.py'



# Parsed testcases at query #54
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_no_scripts. Retrieved 2/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 15/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/cwd'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = "print('test')"
    var_6 = 'obj'
    var_7 = 'name'
    var_8 = 'write'
    var_9 = 'temp_file'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = lambda x, y: var_10
    var_14 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_run_script_with_context_with_python_script. Retrieved 9/15 statements.
# Partially parsed test_run_script_with_context_with_non_python_script. Retrieved 9/15 statements.
# Partially parsed test_run_script_with_context_with_jinja2_env_vars. Retrieved 13/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_8 = 'temp_script.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_8 = 'temp_script.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_jinja2_env_vars'
    var_5 = 'test_project'
    var_6 = 'trim_blocks'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = module_0.run_script_with_context(var_0, var_1, var_10)
    var_12 = 'temp_script.py'



# Parsed testcases at query #58
#--------------------------




def test_case_0():
    var_0 = bool(not 1 != 0)
    assert var_0 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_dir'



# Parsed testcases at query #60
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hook.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/14 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook'
    var_2 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_run_script_with_context_creates_tempfile_with_correct_suffix. Retrieved 6/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #64
#--------------------------




def test_case_0():
    var_0 = bool(not 0 != 0)
    assert var_0 is True



# Parsed testcases at query #65
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #66
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None



# Parsed testcases at query #67
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    var_2 = bool(var_1 == ['hooks/pre-commit'])
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)
    var_2 = len(var_1)
    assert var_2 == 2
    var_3 = 'hooks/pre-commit'
    var_4 = bool('hooks/pre-commit' in var_1)
    assert var_4 is True
    var_5 = 'hooks/pre-commit.sh'
    var_6 = bool('hooks/pre-commit.sh' in var_1)
    assert var_6 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_find_hook_returns_list_when_hook_exists. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre-commit'
    var_2 = '#!/bin/sh\necho "test"'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_find_hook_returns_none_if_no_valid_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_list_of_valid_hooks. Retrieved 8/12 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 7/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 0
    var_6 = var_4[var_5]



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'unsupported-hook'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.another'



# Parsed testcases at query #71
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent-hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 == ['/path/to/hooks/pre-commit'])
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 != ['/path/to/hooks/pre-commit~'])
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 9/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'unsupported-hook'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.another'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/data/fake-repo-pre-prompt'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/fake-repo-pre-prompt-with-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/fake-repo-pre-prompt-failed-hook'
    var_1 = [var_0]



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/tmp'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_run_script_oserror_not_enoexec. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #76
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_script'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #78
#--------------------------

# Failed to parse test_predicate_at_line_21_evaluates_to_false.




# Parsed testcases at query #79
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 11/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = 'pre_gen_project'
    var_2 = '/some/project/dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_run_script_with_context_calls_run_script_with_rendered_template. Retrieved 9/26 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'test_var'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_6 = 'print("{{ test_var }}")'
    var_7 = b'print("test_value")'
    var_8 = '/tmp/temp_script.py'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]



# Parsed testcases at query #82
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = []
    var_8 = lambda _: var_7
    var_9 = module_0.run_hook(var_0, var_1, var_6)
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_paths. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 5/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #84
#--------------------------

# Failed to parse test_oserror_predicate_false.




# Parsed testcases at query #85
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'



# Parsed testcases at query #86
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #87
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #88
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #89
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #90
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 7/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = True
    var_5 = 'Hook failed'
    var_6 = [var_5]
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #92
#--------------------------




def test_case_0():
    var_0 = 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_fails. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'
    var_1 = [var_0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.another'



# Parsed testcases at query #3
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown-hook'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown-hook.py'
    var_1 = 'unknown-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.txt'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'failing_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'failing_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'undefined_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'undefined_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #10
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/tmp'
    var_7 = module_0.run_hook(var_5, var_6, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_hooks_dir_is_directory. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/test/repo'
    var_1 = [var_0]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 9/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'No %s hook found'
    var_8 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = 'project_name'
    var_3 = {}
    var_4 = 'test_project'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_script.sh'
    var_8 = [var_7]
    var_9 = 'echo "{{ cookiecutter.project_name }}"'
    var_10 = 'utf-8'
    var_11 = '.'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_script_with_context_creates_tempfile_with_delete_false. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_6 = False
    var_7 = 'wb'
    var_8 = '.sh'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = []
    var_8 = lambda _: var_7
    var_9 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/tmp/project'
    var_7 = module_0.run_hook(var_5, var_6, var_4)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = [var_0]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_fails. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'
    var_1 = [var_0]



# Parsed testcases at query #21
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_with_undefined'
    var_1 = 'pre_gen_project'
    var_2 = 'project_with_undefined'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_script_with_context_tempfile_delete_false. Retrieved 8/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/current/working/directory'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_script_with_context(var_0, var_1, var_6)



# Parsed testcases at query #23
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py'
    var_1 = 'commit-msg'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown-hook.py'
    var_1 = 'unknown-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.py~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit.txt'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 'test_hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/other-hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 2
    var_8 = 'test_hooks_dir/pre-commit'
    var_9 = 'test_hooks_dir/pre-commit.bak'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 10/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_0]
    var_8 = '#!/bin/bash\necho "Hello {{ cookiecutter.project_name }}"'
    var_9 = 'utf-8'
    var_10 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_11 = [var_0]



# Parsed testcases at query #27
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = False
    var_5 = 'hook failed'
    var_6 = [var_5]
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = 1
    var_4 = os.path.splitext(var_0)[var_3]
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_oserror_with_enoexec. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed, might be an empty file or missing a shebang'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_script_fails_with_exit_status. Retrieved 4/6 statements.
# Partially parsed test_run_script_fails_with_os_error. Retrieved 4/6 statements.
# Partially parsed test_run_script_fails_with_missing_shebang. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'exit status'
    var_5 = bool('exit status' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'error'
    var_5 = bool('error' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script_without_shebang'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'missing a shebang'
    var_5 = bool('missing a shebang' in var_3)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 10/16 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 10/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "pre-push"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "pre-commit"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 'hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "pre-commit"'
    var_3 = '#!/bin/sh\necho "backup"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_5[var_8]
    var_10 = 'hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "pre-commit"'
    var_3 = '#!/bin/sh\necho "pre-commit.sh"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 2
    var_8 = 'hooks_dir/pre-commit'
    var_9 = 'hooks_dir/pre-commit.sh'
    var_10 = (var_8, var_9)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'
    var_1 = [var_0]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'test_hook'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = False
    var_5 = 'Test error'
    var_6 = [var_5]
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #36
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 21 evaluates to False.'
    var_1 = 'Permission denied'
    var_2 = OSError(var_0, var_1)



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




# Parsed testcases at query #42
#--------------------------

# Failed to parse test_predicate_at_line_21_evaluates_to_false.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_absolute_path_to_valid_hook. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 9/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.another'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.sh'
    var_6 = '/test/dir'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 'temp_script.sh'



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = bool(var_0 != 0)
    assert var_1 is True



# Parsed testcases at query #46
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 17 evaluates to False.'
    var_1 = '/some/repo/dir'
    var_2 = 'pre_gen_project'
    var_3 = '/some/project/dir'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = False
    var_10 = bool(not var_9)
    assert var_10 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_valid_hook_paths. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'test_hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/other-hook'



# Parsed testcases at query #48
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_file'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = True
    var_2 = bool(var_1 == var_0)
    assert var_2 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/data/cookiecutter-no-hooks'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/cookiecutter-with-hooks'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'tests/data/cookiecutter-failing-hook'
    var_1 = [var_0]



# Parsed testcases at query #51
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_with_undefined'
    var_1 = 'pre_gen_project'
    var_2 = 'project_with_undefined'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_script_os_error_empty_file. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'empty file or missing a shebang'
    var_5 = bool('empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed'
    var_5 = bool('Hook script failed' in var_3)
    assert var_5 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #54
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #55
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'repo_dir'
    var_6 = 'hook_name'
    var_7 = 'project_dir'
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_4, var_8)
    var_10 = bool(True)
    assert var_10 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'repo_dir'
    var_6 = 'hook_name'
    var_7 = 'project_dir'
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_4, var_8)
    var_10 = bool(True)
    assert var_10 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'repo_dir'
    var_6 = 'hook_name'
    var_7 = 'project_dir'
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_5, var_6, var_7, var_4, var_8)
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #56
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent-hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 == ['/path/to/hooks/pre-commit'])
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 != ['/path/to/hooks/pre-commit~'])
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 != ['/path/to/hooks/post-commit'])
    assert var_3 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/tmp'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.py'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 'temp_script.py'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pre_prompt_hook_returns_original_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = [var_0]



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_exit_status_not_success_raises_exception. Retrieved 2/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #61
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './test_repo'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './test_repo'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './test_repo'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './test_repo'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './test_repo'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #62
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #63
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/9 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/8 statements.
# Partially parsed test_run_script_failed_python_script. Retrieved 3/5 statements.
# Partially parsed test_run_script_os_error_no_exec. Retrieved 4/7 statements.
# Partially parsed test_run_script_os_error_other. Retrieved 4/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = 'No exec'
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = 'Permission denied'
    var_3 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #65
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 12/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = None
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/9 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/8 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/7 statements.
# Partially parsed test_run_script_os_error. Retrieved 4/9 statements.
# Partially parsed test_run_script_generic_os_error. Retrieved 4/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed (exit status: 1)'
    var_5 = bool('Hook script failed (exit status: 1)' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'Executable not found'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'Generic OS error'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'Hook script failed (error: Generic OS error)'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_work_in_context_manager_enters_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/test/directory'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 9/15 statements.


def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = [var_0]
    var_2 = 'pre_gen_project'
    var_3 = '/some/project/dir'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True



# Parsed testcases at query #70
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'



# Parsed testcases at query #71
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #72
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #73
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'test_hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks_dir/post-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'test_hooks_dir/pre-commit'
    var_8 = 'test_hooks_dir/pre-commit.sh'



# Parsed testcases at query #75
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #77
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = len(var_0)
    var_2 = 0
    var_3 = var_1 == var_2
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_find_hook_predicate_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 10/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_hooks_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/post-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 2
    var_8 = 'hooks/pre-commit'
    var_9 = 'hooks/pre-commit.sh'
    var_10 = (var_8, var_9)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 11/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/some/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #82
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #83
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #85
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)



# Parsed testcases at query #86
#--------------------------




def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_hooks. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'



# Parsed testcases at query #88
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = bool(True)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_when_delete_project_on_failure_is_false. Retrieved 9/17 statements.


def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = [var_0]
    var_2 = 'pre_gen_project'
    var_3 = '/some/project/dir'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = False



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = bool(var_2)
    assert var_7 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_with_delete_project_on_failure. Retrieved 11/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #92
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #93
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 12/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = None
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #95
#--------------------------




def test_case_0():
    var_0 = 0



