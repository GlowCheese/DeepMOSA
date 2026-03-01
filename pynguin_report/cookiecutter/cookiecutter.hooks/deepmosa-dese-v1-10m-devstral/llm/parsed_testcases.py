####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'post_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unknown_hook.py'
    var_1 = 'unknown_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py~'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.txt'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_or_no_shebang_script'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 8/20 statements.
# Partially parsed test_run_hook_with_multiple_hooks. Retrieved 11/27 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("test")'
    var_3 = 'pre_gen_project'
    var_4 = 'test'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_4: var_5}

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = 'print("test1")'
    var_4 = 'echo "test2"'
    var_5 = 'pre_gen_project'
    var_6 = 'test'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}
    var_10 = {var_6: var_7}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/15 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = 'temp_script.sh'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 10/12 statements.


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
    var_8 = lambda x: var_7
    var_9 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_run_pre_prompt_hook_no_hooks.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/11 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt'
    var_2 = '#!/bin/sh\necho "test"'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook'
    var_2 = '#!/bin/sh\nexit 1'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_hooks_dir_is_directory. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = True



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_run_pre_prompt_hook_with_no_scripts.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/some/path'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = []
    var_8 = lambda _: var_7
    var_9 = module_0.run_hook(var_0, var_1, var_6)
    var_10 = 'No %s hook found'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_oserror_with_non_enoexec_errno. Retrieved 5/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/path/to/cwd'
    var_2 = 'Permission denied'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = str(var_3)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/fake/repo'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 6/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'



# Parsed testcases at query #22
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = len(var_0)
    var_2 = 0
    var_3 = var_1 == var_2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/5 statements.
# Partially parsed test_find_hook_returns_valid_hook_paths. Retrieved 6/11 statements.
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
    var_8 = 'hooks/pre-commit.sh'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_predicate_at_line_21_evaluates_to_false.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_valid_hook_paths. Retrieved 9/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 7/12 statements.


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
    var_2 = '#!/bin/sh\necho "invalid"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "valid"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'hooks/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "backup"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "valid1"'
    var_3 = '#!/bin/sh\necho "valid2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2



# Parsed testcases at query #32
#--------------------------




import cookiecutter.utils as module_0

def test_case_0():
    var_0 = module_0.work_in()
    assert var_0 is None



# Parsed testcases at query #33
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
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = 'project_name'
    var_3 = {}
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_script.py'
    var_8 = 'print("{{ cookiecutter.project_name }}")'
    var_9 = '.'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file. Retrieved 10/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '#!/bin/bash\necho "Hello {{ cookiecutter.project_name }}"'
    var_8 = 'utf-8'
    var_9 = module_0.run_script_with_context(var_0, var_1, var_6)



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_delete_project_on_failure. Retrieved 7/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = {}
    var_4 = True
    var_5 = 'Hook failed'
    var_6 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 8/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/repo/dir'
    var_1 = 'pre_gen_project'
    var_2 = '/project/dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pre_prompt_hook_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'cookiecutter'



# Parsed testcases at query #41
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.sh'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_with_delete_project_on_failure_false. Retrieved 9/12 statements.


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
    var_8 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #43
#--------------------------




def test_case_0():
    var_0 = 1



# Parsed testcases at query #44
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '.'
    var_1 = 'pre_gen_project'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #46
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_run_script_failed_exit_status. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_no_exec. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 4/6 statements.


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
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_or_no_shebang_script'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)



# Parsed testcases at query #48
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
    var_6 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_oserror_with_enonexec_raises_failedhookeception. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Executable not found'
    var_1 = '/path/to/script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #50
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_tempfile_delete_false_mode_wb_suffix_extension. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/cwd'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'utf-8'
    var_8 = False
    var_9 = 'wb'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_uses_work_in_context_manager. Retrieved 10/13 statements.


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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test error'
    var_1 = 'repo_dir'
    var_2 = 'hook_name'
    var_3 = 'project_dir'
    var_4 = {}
    var_5 = False
    var_6 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/14 statements.


def test_case_0():
    var_0 = '/fake/script.sh'
    var_1 = '/fake/cwd'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'echo "{{ cookiecutter.name }}"'
    var_8 = 'utf-8'



# Parsed testcases at query #56
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'project_output'
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
    var_2 = 'project_output'
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
    var_2 = 'project_output'
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
    var_2 = 'project_output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_predicate_at_line_21. Retrieved 3/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pre_prompt*'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pre_prompt_hook_no_scripts. Retrieved 3/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/dir'
    var_1 = 'pre_prompt'
    var_2 = module_0.find_hook(var_1)



# Parsed testcases at query #60
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './tests/mock_repo'
    var_1 = 'pre_gen_project'
    var_2 = './tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './tests/mock_repo'
    var_1 = 'pre_gen_project'
    var_2 = './tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = './tests/mock_repo'
    var_1 = 'pre_gen_project'
    var_2 = './tests/mock_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




# Parsed testcases at query #62
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate. Retrieved 7/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'test_project'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_empty_file. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/test_script.sh'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)



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
    var_0 = 'pre-commit.txt'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #4
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit.py'
    var_1 = 'pre_commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_hooks_dir_exists. Retrieved 2/3 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.find_hook(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pre_prompt_hook_no_scripts. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/13 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_exception. Retrieved 12/19 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 12/19 statements.


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
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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
    var_9 = 'test error'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_11 = "Stopping generation because %s hook script didn't exit successfully"

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
    var_9 = 'test error'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_11 = "Stopping generation because %s hook script didn't exit successfully"



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 11/14 statements.
# Partially parsed test_run_hook_with_no_hook. Retrieved 8/10 statements.


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
    var_10 = [call(script, var_1, var_6) for script in var_8]

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



# Parsed testcases at query #14
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
    var_1 = 'failing_hook'
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
    var_1 = 'failing_hook'
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
    var_1 = 'undefined_hook'
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
    var_1 = 'undefined_hook'
    var_2 = './test_project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/test-data/cookiecutter-no-hook'

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter-with-hook'

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter-failed-hook'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/data/test-template-no-hooks'

def test_case_0():
    var_0 = 'tests/data/test-template-with-hooks'

def test_case_0():
    var_0 = 'tests/data/test-template-with-failing-hook'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 12/17 statements.
# Partially parsed test_run_hook_with_invalid_hook. Retrieved 8/11 statements.
# Partially parsed test_run_hook_with_backup_file. Retrieved 8/11 statements.
# Partially parsed test_run_hook_with_multiple_hooks. Retrieved 14/23 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks'
    var_1 = True
    var_2 = '/tmp/hooks/pre_gen_project.py'
    var_3 = 'print("Hello")'
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = module_0.run_hook(var_4, var_5, var_10)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks'
    var_1 = True
    var_2 = '/tmp/hooks/invalid_hook.py'
    var_3 = 'print("Hello")'
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project'
    var_6 = {}
    var_7 = module_0.run_hook(var_4, var_5, var_6)
    assert var_7 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks'
    var_1 = True
    var_2 = '/tmp/hooks/pre_gen_project.py~'
    var_3 = 'print("Hello")'
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project'
    var_6 = {}
    var_7 = module_0.run_hook(var_4, var_5, var_6)
    assert var_7 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks'
    var_1 = True
    var_2 = '/tmp/hooks/pre_gen_project.py'
    var_3 = 'print("Hello")'
    var_4 = '/tmp/hooks/pre_gen_project.sh'
    var_5 = 'echo "Hello"'
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = module_0.run_hook(var_6, var_7, var_12)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 5/17 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 5/13 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'test_repo'

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = '#!/bin/sh\necho "test"'
    var_4 = 493

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'invalid_hook'
    var_3 = '#!/bin/sh\necho "test"'
    var_4 = 493

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = '#!/bin/sh\nexit 1'
    var_4 = 493



# Parsed testcases at query #2
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
    var_8 = 'hooks/pre-commit.sh'



# Parsed testcases at query #4
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
    var_0 = 'empty_script.sh'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/path/to/repo'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_7 = module_0.run_hook(var_0, var_1, var_6)
    assert var_7 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/hooks/pre_gen_project.py'
    var_8 = [var_7]
    var_9 = module_0.find_hook(var_0)
    var_10 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hooks. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_failed_script. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'



# Parsed testcases at query #9
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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #10
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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'no_hook_repo'
    var_1 = 'non_existent_hook'
    var_2 = 'no_hook_project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 8/11 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/fake/dir/hook.sh'
    var_7 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'cookiecutter'



# Parsed testcases at query #13
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.run_hook(var_0, var_1, var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_hooks_dir_is_not_a_directory. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #15
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 15/26 statements.


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
    var_10 = 'utf-8'
    var_11 = False
    var_12 = 'wb'
    var_13 = '.sh'
    var_14 = 'temp_script.sh'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_predicate_true. Retrieved 2/3 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook_name'
    var_1 = module_0.find_hook(var_0)



# Parsed testcases at query #18
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
    var_0 = 'pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/13 statements.


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
    var_10 = 'No %s hook found'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'echo "Hello {{ cookiecutter.project_name }}"'
    var_8 = 'utf-8'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_oserror_with_enexec_errno_raises_failed_hook_exception. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_hooks_dir_is_not_directory. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nonexistent_dir'



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_repo'



# Parsed testcases at query #26
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook_name'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/6 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'tests/mocks/pre-and-post-hooks'

def test_case_0():
    var_0 = 'tests/mocks/pre-and-post-hooks'

def test_case_0():
    var_0 = 'tests/mocks/pre-and-post-hooks'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_delete_project_on_failure. Retrieved 11/15 statements.


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
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_predicate_at_line_21_evaluates_to_false.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate_false. Retrieved 7/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 20 evaluates to False.'
    var_1 = 'test_repo'
    var_2 = 'test_hook'
    var_3 = 'test_project'
    var_4 = {}
    var_5 = False
    var_6 = module_0.run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)



# Parsed testcases at query #33
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
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #34
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existent_hook'
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



# Parsed testcases at query #35
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test_dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 14 evaluates to False.'
    var_1 = 'test_script.py'
    var_2 = '/test/cwd'
    var_3 = 'test'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.run_script_with_context(var_1, var_2, var_5)



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_work_in_context_manager_is_used.




# Parsed testcases at query #38
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hooks. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 1/5 statements.
# Partially parsed test_run_pre_prompt_hook_failed_execution. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-failing-hook'



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None
    var_4 = 'post_gen_project'
    var_5 = {}
    var_6 = module_0.run_hook(var_4, var_1, var_5)
    assert var_6 is None
    var_7 = 'pre_hook'
    var_8 = {}
    var_9 = module_0.run_hook(var_7, var_1, var_8)
    assert var_9 is None
    var_10 = 'post_hook'
    var_11 = {}
    var_12 = module_0.run_hook(var_10, var_1, var_11)
    assert var_12 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'pre_gen_project'
    var_6 = '/path/to/project'
    var_7 = module_0.run_hook(var_5, var_6, var_4)
    assert var_7 is None
    var_8 = 'post_gen_project'
    var_9 = module_0.run_hook(var_8, var_6, var_4)
    assert var_9 is None
    var_10 = 'pre_hook'
    var_11 = module_0.run_hook(var_10, var_6, var_4)
    assert var_11 is None
    var_12 = 'post_hook'
    var_13 = module_0.run_hook(var_12, var_6, var_4)
    assert var_13 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/8 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/7 statements.
# Partially parsed test_run_script_failed_hook_exception. Retrieved 4/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = False
    var_3 = module_0.run_script(var_0, var_1)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = [var_0]
    var_3 = False
    var_4 = module_0.run_script(var_0, var_1)
    assert var_4 is None

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

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_valid_hook_path. Retrieved 6/13 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_ignores_unsupported_hooks. Retrieved 6/11 statements.


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
    var_5 = 'test_hooks_dir/unsupported-hook'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'project_output'
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
    var_2 = 'project_output'
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
    var_2 = 'project_output'
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
    var_2 = 'project_output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'undefined_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'project_output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/fake/repo'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #10
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_hook_no_scripts_found. Retrieved 11/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/path/to/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = []
    var_8 = lambda _: var_7
    var_9 = module_0.run_hook(var_0, var_1, var_6)
    var_10 = 'No %s hook found'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_hooks_dir_is_not_a_directory. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_hook'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_hooks_dir_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/empty_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid_script.py'
    var_1 = '/working/directory'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook_name'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/path/to/repo'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 4/10 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 5/9 statements.
# Partially parsed test_run_script_failed_python_script. Retrieved 4/11 statements.
# Partially parsed test_run_script_failed_non_python_script. Retrieved 5/10 statements.
# Partially parsed test_run_script_os_error. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_no_exec. Retrieved 4/6 statements.


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
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = [var_0]
    var_3 = False
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)



# Parsed testcases at query #22
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
    var_0 = 'pre-commit.sh'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/pre-commit.py'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #23
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
    var_2 = 'project_to_delete'
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
    var_2 = 'project_to_keep'
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
    var_2 = 'project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_hooks_dir_is_directory. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = 'hooks'



# Parsed testcases at query #25
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/directory'
    var_2 = module_0.run_script(var_0, var_1)
    assert var_2 is None

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



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_pre_prompt_hook_no_scripts.




# Parsed testcases at query #27
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_run_pre_prompt_hook_no_hook. Retrieved 1/3 statements.
# Partially parsed test_run_pre_prompt_hook_with_hook. Retrieved 1/4 statements.
# Partially parsed test_run_pre_prompt_hook_failed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'tests/fake-repo-pre'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-with-hook'

def test_case_0():
    var_0 = 'tests/fake-repo-pre-failed-hook'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_oserror_with_enonexec. Retrieved 4/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/test/dir'
    var_2 = 'Test error'
    var_3 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    var_0 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '/tmp'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'echo "{{ project_name }}"'
    var_6 = 'utf-8'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_work_in_context_manager_is_used. Retrieved 6/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_with_delete_project_on_failure_false. Retrieved 11/15 statements.


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
    var_9 = 'Hook failed'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pre_prompt_hook_no_scripts. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_dir'



# Parsed testcases at query #37
#--------------------------




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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_oserror_with_enoexec_raises_failed_hook_exception. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Executable format error'
    var_1 = 'dummy_script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_none_when_only_backup_files_exist. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_ignores_unsupported_hook. Retrieved 6/11 statements.
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
    var_0 = 'hooks_with_backup'
    var_1 = True
    var_2 = '#!/bin/sh\necho "backup"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks_with_backup/pre-commit~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_with_valid'
    var_1 = True
    var_2 = '#!/bin/sh\necho "valid"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'hooks_with_valid/pre-commit'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_with_unsupported'
    var_1 = True
    var_2 = '#!/bin/sh\necho "unsupported"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    assert var_4 is None
    var_5 = 'hooks_with_unsupported/unsupported-hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks_with_multiple'
    var_1 = True
    var_2 = '#!/bin/sh\necho "first"'
    var_3 = '#!/bin/sh\necho "second"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks_with_multiple/pre-commit'
    var_8 = 'hooks_with_multiple/pre-commit.sh'



# Parsed testcases at query #40
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
    var_8 = False
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
    var_0 = 'undefined_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'undefined_project'
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



# Parsed testcases at query #41
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script_fail.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script_empty'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script_no_shebang'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 12/25 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_script.sh'
    var_6 = '/tmp'
    var_7 = module_0.run_script_with_context(var_5, var_6, var_4)
    var_8 = False
    var_9 = 'wb'
    var_10 = '.sh'
    var_11 = '/tmp/temp_script.sh'



# Parsed testcases at query #43
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_matching_scripts. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = True
    var_3 = module_0.find_hook(var_0, var_1)
    assert var_3 is None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #46
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_find_hook_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'hooks'



# Parsed testcases at query #49
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_find_hook_returns_none_when_no_valid_hooks. Retrieved 4/6 statements.
# Partially parsed test_find_hook_returns_none_when_no_matching_hooks. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_list_with_valid_hook. Retrieved 9/15 statements.
# Partially parsed test_find_hook_ignores_backup_files. Retrieved 6/11 statements.
# Partially parsed test_find_hook_returns_multiple_valid_hooks. Retrieved 9/21 statements.


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
    assert var_4 is None
    var_5 = 'hooks/other-hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = '#!/bin/sh\necho "test"'
    var_3 = 'pre-commit'
    var_4 = module_0.find_hook(var_3, var_2)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'hooks/pre-commit'

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
    var_2 = '#!/bin/sh\necho "test1"'
    var_3 = '#!/bin/sh\necho "test2"'
    var_4 = 'pre-commit'
    var_5 = module_0.find_hook(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'hooks/pre-commit'
    var_8 = 'hooks/pre-commit.sh'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_run_script_os_error_empty_file. Retrieved 4/6 statements.
# Partially parsed test_run_script_os_error_general. Retrieved 4/6 statements.


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
    var_3 = str(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_script.py'
    var_1 = '/test/dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = str(var_2)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_predicate. Retrieved 11/13 statements.


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
    assert var_8 is True
    var_9 = 'Hook failed'
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_work_in_context_manager_changes_directory.




# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------




def test_case_0():
    var_0 = 1



