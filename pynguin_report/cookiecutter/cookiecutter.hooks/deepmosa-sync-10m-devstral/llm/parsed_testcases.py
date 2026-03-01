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



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/13 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 11/18 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 11/18 statements.


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
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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
    var_8 = False
    var_9 = 'Undefined variable'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_hooks_dir_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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
    var_0 = 'no_shebang_script'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 4/17 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 4/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test_repo'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'invalid_hook'
    var_3 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_ignores_non_matching_hooks. Retrieved 5/8 statements.


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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_predicate_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = True



# Parsed testcases at query #10
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #11
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
    var_0 = 'invalid-hook.py'
    var_1 = 'invalid-hook'
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



# Parsed testcases at query #12
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mocks/pretend-repo-1'
    var_1 = 'pre_gen_project'
    var_2 = 'tests/mocks/pretend-project-1'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mocks/pretend-repo-1'
    var_1 = 'pre_gen_project'
    var_2 = 'tests/mocks/pretend-project-1'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mocks/pretend-repo-1'
    var_1 = 'pre_gen_project'
    var_2 = 'tests/mocks/pretend-project-1'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #13
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
    var_0 = 'pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #14
#--------------------------




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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = [var_0]



# Parsed testcases at query #16
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'post_gen_project'
    var_6 = '/tmp'
    var_7 = module_0.run_hook(var_5, var_6, var_4)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = '/fake/path'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #20
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/path/to/project'
    var_2 = 'pre_gen_project'
    var_3 = module_0.find_hook(var_2)
    var_4 = []
    var_5 = var_3 == var_4
    var_6 = module_0.run_hook(var_2, var_1, var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_hook_empty_directory. Retrieved 4/6 statements.
# Partially parsed test_find_hook_no_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_with_matching_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 6/11 statements.
# Partially parsed test_find_hook_with_multiple_matching_hooks. Retrieved 9/21 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks'
    var_1 = True
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks/other-hook.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks'
    var_1 = True
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 'test_hooks/pre-commit.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks'
    var_1 = True
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'test_hooks/pre-commit.py~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hooks'
    var_1 = True
    var_2 = '#!/usr/bin/env python\nprint("test1")'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 2
    var_8 = 'test_hooks/pre-commit.py'
    var_9 = 'test_hooks/pre-commit.sh'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_run_script_with_context_success. Retrieved 7/12 statements.
# Partially parsed test_run_script_with_context_failure. Retrieved 5/11 statements.
# Partially parsed test_run_script_with_context_empty_file. Retrieved 5/11 statements.
# Partially parsed test_run_script_with_context_python_script. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = [var_0]
    var_2 = 'echo "Hello {{ name }}"'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = '.'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = [var_0]
    var_2 = 'exit 1'
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = '.'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = [var_0]
    var_2 = ''
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = '.'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = [var_0]
    var_2 = 'print("Hello {{ name }}")'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = '.'
    var_8 = [var_7]



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/some/repo'
    var_1 = [var_0]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_list_of_valid_hooks. Retrieved 9/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 7/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '# invalid hook'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '# valid hook'
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
    var_2 = '# backup hook'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '# valid hook'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 0
    var_6 = var_4[var_5]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pre_prompt_hook_returns_original_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = [var_0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_extension. Retrieved 9/11 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/working/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = False
    var_8 = 'wb'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/10 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/9 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/11 statements.
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
    var_4 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed (error: '
    var_5 = bool('Hook script failed (error: ' in var_3)
    assert var_5 is True



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate. Retrieved 11/18 statements.


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
    var_13 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 6/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = ()
    var_3 = 'Executable not found'
    var_4 = None
    var_5 = module_0.run_script(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/7 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = bool(var_0 == var_2)
    assert var_3 is True



# Parsed testcases at query #36
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
    var_2 = bool(var_1 == ['hooks/pre-commit.sh'])
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
    var_3 = 'hooks/pre-commit.sh'
    var_4 = bool('hooks/pre-commit.sh' in var_1)
    assert var_4 is True
    var_5 = 'hooks/pre-commit.bash'
    var_6 = bool('hooks/pre-commit.bash' in var_1)
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #38
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    assert var_5 is None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 11/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = 'test_project'
    var_4 = 'test_repo'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_script.sh'
    var_8 = '/tmp'
    var_9 = module_0.run_script_with_context(var_7, var_8, var_6)
    var_10 = 'temp_script.sh'



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_with_delete_project_on_failure_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = '/fake/repo/dir'
    var_1 = [var_0]
    var_2 = 'pre_gen_project'
    var_3 = '/fake/project/dir'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = [var_11]



# Parsed testcases at query #42
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #43
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'failing_hook'
    var_2 = 'tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'failing_hook'
    var_2 = 'tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'nonexistent_hook'
    var_2 = 'tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'print("Hello {{ cookiecutter.project_name }}")'
    var_8 = 'print("Hello test")'
    var_9 = module_0.run_script_with_context(var_0, var_1, var_6)
    var_10 = 'utf-8'
    var_11 = 'temp_script.py'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 4/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'invalid_hook'
    var_4 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 11/14 statements.


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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_oserror_with_enexec_errno. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Exec format error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)
    var_3 = 'Hook script failed, might be an empty file or missing a shebang'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 6/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #49
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 14 evaluates to False.'
    var_1 = 'test_script.py'
    var_2 = [var_1]
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = 'cookiecutter'
    var_6 = '_jinja2_env_vars'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_script_with_context_with_python_script. Retrieved 7/13 statements.
# Partially parsed test_run_script_with_context_with_shell_script. Retrieved 7/13 statements.
# Partially parsed test_run_script_with_context_creates_jinja_env. Retrieved 6/13 statements.
# Partially parsed test_run_script_with_context_renders_template. Retrieved 7/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_script.py'
    var_4 = '/tmp'
    var_5 = module_0.run_script_with_context(var_3, var_4, var_2)
    var_6 = 'temp_script.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_script.sh'
    var_4 = '/tmp'
    var_5 = module_0.run_script_with_context(var_3, var_4, var_2)
    var_6 = 'temp_script.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_script.py'
    var_4 = '/tmp'
    var_5 = module_0.run_script_with_context(var_3, var_4, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_script.py'
    var_4 = '/tmp'
    var_5 = module_0.run_script_with_context(var_3, var_4, var_2)
    var_6 = b'print("test_project")'



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_work_in_context_manager_is_used.




# Parsed testcases at query #53
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook_name'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #54
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
    var_0 = 'empty_script'
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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = bool(var_0 != 0)
    assert var_1 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = False
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_tempfile_suffix_matches_extension. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = False
    var_4 = 'wb'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.bat'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



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
    var_1 = 'nonexistent_dir'
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
    var_2 = bool(var_1 == ['hooks/pre-commit.sh'])
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
    var_2 = bool(var_1 == ['/abs/path/pre-commit.sh'])
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'invalid-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported-hook'
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
    var_0 = 'pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_with_valid_script. Retrieved 10/14 statements.
# Partially parsed test_run_hook_with_invalid_script. Retrieved 11/16 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
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
    var_5 = 'echo "Hello {{ cookiecutter.project_name }}"'
    var_6 = '/tmp/test_hook.sh'
    var_7 = 493
    var_8 = '/tmp'
    var_9 = module_0.run_hook(var_2, var_8, var_4)
    assert var_9 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'invalid_command'
    var_6 = '/tmp/invalid_hook.sh'
    var_7 = 493
    var_8 = 'invalid'
    var_9 = '/tmp'
    var_10 = module_0.run_hook(var_8, var_9, var_4)



# Parsed testcases at query #7
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
    var_7 = 'invalid-hook'
    var_8 = module_0.valid_hook(var_3, var_7)
    assert var_8 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 10/17 statements.
# Partially parsed test_run_hook_with_invalid_hook. Retrieved 6/13 statements.
# Partially parsed test_run_hook_with_backup_file. Retrieved 6/13 statements.
# Partially parsed test_run_hook_with_multiple_scripts. Retrieved 12/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'print("Hello")'
    var_3 = 'pre_gen_project'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'hooks/pre_gen_project.py'

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'invalid'
    var_3 = 'invalid_hook'
    var_4 = {}
    var_5 = 'hooks/invalid_hook.txt'

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'print("Backup")'
    var_3 = 'pre_gen_project'
    var_4 = {}
    var_5 = 'hooks/pre_gen_project.py~'

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'print("First")'
    var_3 = 'echo "Second"'
    var_4 = 'pre_gen_project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'hooks/pre_gen_project.py'
    var_11 = 'hooks/pre_gen_project.sh'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 3/5 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 4/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'print("Hello")'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'hooks'
    var_2 = 'invalid_hook.py'
    var_3 = 'print("Hello")'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'import sys; sys.exit(1)'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_scripts. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 10/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.sh'
    var_6 = '/test/dir'
    var_7 = [var_5]
    var_8 = 'echo "{{ cookiecutter.project_name }}"'
    var_9 = 'utf-8'
    var_10 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tempfile_suffix_matches_script_extension. Retrieved 2/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_script.py'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_hook_empty_dir. Retrieved 4/6 statements.
# Partially parsed test_find_hook_invalid_hook_file. Retrieved 6/11 statements.
# Partially parsed test_find_hook_valid_hook_file. Retrieved 7/13 statements.
# Partially parsed test_find_hook_backup_file. Retrieved 6/11 statements.
# Partially parsed test_find_hook_multiple_valid_files. Retrieved 9/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_dir'
    var_1 = True
    var_2 = 'pre-commit'
    var_3 = module_0.find_hook(var_2, var_0)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/invalid_hook.txt'

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
    var_7 = 'hooks/pre-commit'
    var_8 = var_4[0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = 'hooks/pre-commit'
    var_9 = var_5[0]
    var_10 = 'hooks/pre-commit.bak'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_hook_returns_none_for_empty_directory. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_none_for_no_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 9/19 statements.


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
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks_dir/other-hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = 'hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks_dir/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_dir'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks_dir/pre-commit'
    var_8 = 'hooks_dir/pre-commit.sh'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 8/11 statements.
# Partially parsed test_run_hook_with_multiple_hooks. Retrieved 10/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/path/to/hook_script.py'
    var_7 = {var_2: var_3}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/path/to/hook1.py'
    var_7 = {var_2: var_3}
    var_8 = '/path/to/hook2.sh'
    var_9 = {var_2: var_3}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_none_when_only_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_matching_hooks. Retrieved 9/17 statements.


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
    var_0 = 'backup_hooks_dir'
    var_1 = True
    var_2 = "#!/bin/sh\necho 'backup'"
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'backup_hooks_dir/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hooks_dir'
    var_1 = True
    var_2 = "#!/bin/sh\necho 'valid'"
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 'valid_hooks_dir/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported_hooks_dir'
    var_1 = True
    var_2 = "#!/bin/sh\necho 'unsupported'"
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'unsupported_hooks_dir/unsupported-hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'multiple_hooks_dir'
    var_1 = True
    var_2 = "#!/bin/sh\necho 'first'"
    var_3 = "#!/bin/sh\necho 'second'"
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = len(var_5)
    assert var_7 == 2
    var_8 = 'multiple_hooks_dir/pre-commit'
    var_9 = 'multiple_hooks_dir/pre-commit.sh'



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'valid_project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'invalid_project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo_with_undefined_var'
    var_1 = 'pre_gen_project'
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_hook_successful_execution. Retrieved 10/14 statements.
# Partially parsed test_run_hook_with_invalid_script. Retrieved 10/13 statements.
# Partially parsed test_run_hook_with_jinja_template. Retrieved 10/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.py'
    var_1 = [var_0]
    var_2 = '#!/usr/bin/env python\nprint("Hello")'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'
    var_9 = '/tmp'
    var_10 = module_0.run_hook(var_8, var_9, var_7)
    var_11 = [var_0]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.sh'
    var_1 = [var_0]
    var_2 = 'invalid command'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'
    var_9 = '/tmp'
    var_10 = module_0.run_hook(var_8, var_9, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.py'
    var_1 = [var_0]
    var_2 = 'print("{{ cookiecutter.project_name }}")'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'
    var_9 = '/tmp'
    var_10 = module_0.run_hook(var_8, var_9, var_7)
    var_11 = [var_0]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/9 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/8 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/11 statements.
# Partially parsed test_run_script_os_error_enoexec. Retrieved 4/6 statements.
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
    var_4 = 'Hook script failed (exit status: 1)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'
    var_5 = bool('Hook script failed, might be an empty file or missing a shebang' in var_3)
    assert var_5 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)
    var_4 = 'Hook script failed (error:'
    var_5 = bool('Hook script failed (error:' in var_3)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hook.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/16 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook'
    var_2 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = 'project_name'
    var_3 = {}
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'echo "{{ cookiecutter.project_name }}"'
    var_8 = '.sh'
    var_9 = 'utf-8'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pre_prompt_hook_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'cookiecutter'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 12/17 statements.


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
    var_8 = True
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = None
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_hook_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #29
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 10/14 statements.


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
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_work_in_context_manager_changes_and_restores_directory.




# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_script_with_context_with_python_script. Retrieved 15/35 statements.
# Partially parsed test_run_script_with_context_with_shell_script. Retrieved 15/35 statements.


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
    var_8 = 'utf-8'
    var_9 = 'print("{{ cookiecutter.project_name }}")'
    var_10 = False
    var_11 = 'wb'
    var_12 = '.py'
    var_13 = 'print("test_project")'
    var_14 = 'temp_script.py'

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
    var_8 = 'utf-8'
    var_9 = 'echo "{{ cookiecutter.project_name }}"'
    var_10 = False
    var_11 = 'wb'
    var_12 = '.sh'
    var_13 = 'echo "test_project"'
    var_14 = 'temp_script.sh'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #35
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
    var_0 = 'test_script.py'
    var_1 = module_0.run_script(var_0)
    var_2 = bool(True)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'failing_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_find_hook_returns_absolute_paths. Retrieved 3/5 statements.


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
    var_3 = bool(var_2 == ['/path/to/hooks/pre-commit'])
    assert var_3 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_oserror_without_enoexec_raises_failedhookeception. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Permission denied'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_run_hook_from_repo_dir_predicate.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_false_when_delete_project_on_failure_is_false. Retrieved 12/18 statements.


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
    var_11 = None
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = bool(var_2)
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_oserror_with_non_enoexec_errno. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = 'Permission denied'
    var_4 = 'Hook script failed (error: [Errno 13] Permission denied)'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_oserror_predicate_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'Permission denied'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.sh'
    var_6 = 'echo "test"'
    var_7 = 'utf-8'



# Parsed testcases at query #46
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_suffix. Retrieved 9/12 statements.


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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'fake_hook'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = False
    var_5 = 'test'
    var_6 = [var_5]
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/5 statements.
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



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_work_in_context_manager_is_used.




# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hooks. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 4/12 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 4/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'invalid_hook'
    var_4 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = [var_0]
    var_2 = 'hooks'
    var_3 = 'pre_prompt'
    var_4 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_21. Retrieved 3/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 4/17 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'repo'

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hooks'
    var_2 = 'invalid_hook'
    var_3 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #54
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 9/12 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 15/22 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 15/22 statements.


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
    var_8 = {var_3: var_4}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = [var_0]
    var_2 = 'repo_dir'
    var_3 = 'hook_name'
    var_4 = 'project_dir'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_2, var_3, var_4, var_7, var_8)
    var_10 = 'repo_dir'
    var_11 = 'hook_name'
    var_12 = 'project_dir'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'repo_dir'
    var_2 = 'hook_name'
    var_3 = 'project_dir'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = 'repo_dir'
    var_10 = 'hook_name'
    var_11 = 'project_dir'
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/11 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/10 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/10 statements.
# Partially parsed test_run_script_os_error_no_exec. Retrieved 4/10 statements.
# Partially parsed test_run_script_os_error_other. Retrieved 4/10 statements.
# Partially parsed test_run_script_windows_platform. Retrieved 4/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = [var_0]
    var_4 = False

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
    var_2 = 'No exec'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'Hook script failed, might be an empty file or missing a shebang'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'Permission denied'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'Hook script failed (error: [Errno 13] Permission denied)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate. Retrieved 11/13 statements.


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
    assert var_8 is True
    var_9 = 'Hook failed'
    var_10 = [var_9]
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #60
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #61
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #62
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.sh'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #63
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #64
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = 'some_hook'
    var_2 = '/some/project/dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook(var_1, var_2, var_3)
    var_6 = bool(not var_4)
    assert var_6 is True



# Parsed testcases at query #65
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



