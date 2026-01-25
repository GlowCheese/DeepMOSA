####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hooks/pre-commit'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hooks/unsupported-hook'
    var_1 = 'unsupported-hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hooks/commit-msg'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hooks/pre-commit~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hooks/commit-msg~'
    var_1 = 'pre-commit'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #2
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid_script.py'
    var_1 = module_0.run_script(var_0)



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 8/18 statements.
# Partially parsed test_run_hook_with_no_hook_found. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'pre_gen_project'
    var_5 = '/tmp/project/hooks/pre_gen_project.sh'
    var_6 = '/tmp/tempfile.sh'
    var_7 = module_0.run_hook(var_4, var_3, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'pre_gen_project'
    var_5 = module_0.run_hook(var_4, var_3, var_2)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook.py'
    var_1 = 'valid_hook'
    var_2 = 'valid_hook'
    var_3 = {var_2}
    var_4 = module_0.valid_hook(var_0, var_1)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 6/12 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 6/12 statements.
# Partially parsed test_run_pre_prompt_hook_with_no_hooks_dir. Retrieved 3/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_empty_hooks_dir. Retrieved 4/7 statements.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 7/14 statements.


import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'print("Pre-prompt hook executed")'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'invalid_hook.py'
    var_3 = 'print("Invalid hook executed")'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = module_0.run_pre_prompt_hook(var_0)
    var_3 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'exit(1)'
    var_4 = None
    var_5 = module_0.run_pre_prompt_hook(var_0)
    assert var_5 is None
    var_6 = module_1.rmtree(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_hook_with_valid_hook_and_directory. Retrieved 4/7 statements.
# Partially parsed test_find_hook_with_valid_hook_and_directory_multiple_files. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fixtures/hooks'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    var_3 = 'pre_gen_project.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fixtures/hooks_multiple'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    var_3 = 'pre_gen_project.py'
    var_4 = 'pre_gen_project.sh'
    var_5 = sorted(var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fixtures/hooks'
    var_1 = 'invalid_hook'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fixtures/invalid_hooks_dir'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fixtures/hooks_with_backup'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    var_3 = 'pre_gen_project.py'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_run_hook_with_valid_hook. Retrieved 7/10 statements.
# Partially parsed test_run_hook_with_empty_hooks_dir. Retrieved 8/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'pre_gen_project'
    var_5 = b'#!/usr/bin/env python\nprint("hook executed")'
    var_6 = module_0.run_hook(var_4, var_3, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'invalid_hook'
    var_5 = module_0.run_hook(var_4, var_3, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = True
    var_7 = module_0.run_hook(var_4, var_3, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'pre_gen_project'
    var_5 = module_0.run_hook(var_4, var_3, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/project'
    var_4 = 'unsupported_hook'
    var_5 = module_0.run_hook(var_4, var_3, var_2)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 6/15 statements.
# Partially parsed test_find_hook_with_invalid_hook. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = True
    var_4 = ''
    var_5 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = True
    var_4 = ''
    var_5 = module_0.find_hook(var_0, var_1)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py~'
    var_3 = True
    var_4 = ''
    var_5 = module_0.find_hook(var_0, var_1)
    assert var_5 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existing_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = True
    var_3 = module_0.find_hook(var_0, var_1)
    assert var_3 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 6/15 statements.
# Partially parsed test_find_hook_with_invalid_hook. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 9/24 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks'
    var_2 = True
    var_3 = 'test'
    var_4 = module_0.find_hook(var_0, var_1)
    var_5 = 'pre-commit.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks'
    var_2 = True
    var_3 = 'test'
    var_4 = module_0.find_hook(var_0, var_1)
    assert var_4 is None
    var_5 = 'post-commit.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks'
    var_2 = True
    var_3 = 'test'
    var_4 = module_0.find_hook(var_0, var_1)
    assert var_4 is None
    var_5 = 'pre-commit.sh~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/non_existing_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks'
    var_2 = True
    var_3 = 'test'
    var_4 = 'test'
    var_5 = module_0.find_hook(var_0, var_1)
    var_6 = set(var_5)
    var_7 = 'pre-commit.sh'
    var_8 = 'pre-commit.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_invalid_hook_name. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_non_existent_hooks_dir. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_empty_hooks_dir. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'invalid_hook'

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
    var_0 = 'non_existent_hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'



# Parsed testcases at query #13
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/test/dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/13 statements.
# Failed to parse test_run_pre_prompt_hook_with_no_hook.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Hello, World!")'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'exit(1)'



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_script_with_context_creates_and_runs_rendered_script. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'Test Project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = "echo '{{ cookiecutter.project_name }}'"
    var_6 = 'test_script.sh'
    var_7 = 'utf-8'
    var_8 = '.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/12 statements.
# Partially parsed test_find_hook_with_invalid_hook_name. Retrieved 3/8 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/10 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 3/8 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'test_hook.py'
    var_2 = ''
    var_3 = 'test_hook'

def test_case_0():
    var_0 = 'invalid_hook.py'
    var_1 = ''
    var_2 = 'test_hook'

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'test_hook.py~'
    var_2 = ''
    var_3 = 'test_hook'

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = ''
    var_2 = 'unsupported_hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'test_hook'
    var_1 = 'test_hook.py'
    var_2 = 'test_hook.sh'
    var_3 = ''
    var_4 = ''
    var_5 = 'test_hook'



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks'
    var_2 = 'tests/test_hooks/pre-commit'
    var_3 = [var_2]
    var_4 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid-hook'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks_with_backup'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported-hook'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/test_hooks_multiple'
    var_2 = 'tests/test_hooks_multiple/pre-commit'
    var_3 = 'tests/test_hooks_multiple/pre-commit.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.find_hook(var_0, var_1)
    var_6 = sorted(var_5)
    var_7 = sorted(var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some/dir'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 7/12 statements.
# Partially parsed test_run_pre_prompt_hook_without_hook. Retrieved 3/4 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 6/11 statements.


import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = f'{var_0}/hooks'
    var_2 = f'{var_0}/hooks/pre_prompt.py'
    var_3 = 'print("Hello, World!")'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_1.rmtree(var_0)
    var_6 = module_1.rmtree(var_4)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = f'{var_0}/hooks'
    var_2 = f'{var_0}/hooks/pre_prompt.py'
    var_3 = 'exit(1)'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_1.rmtree(var_0)



# Parsed testcases at query #22
#--------------------------




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
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #23
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #24
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_run_script_with_context_creates_non_deletable_temp_file. Retrieved 7/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_6 = '.py'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_script_oserror_not_enoexec. Retrieved 1/4 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #27
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_executable_file.txt'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = OSError()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_run_script_with_context_creates_temporary_file_with_correct_extension. Retrieved 9/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/path/to/cwd'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_7 = '.py'
    var_8 = None



# Parsed testcases at query #30
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'some_hook'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'some_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_run_script_with_context_creates_temporary_file_with_correct_extension. Retrieved 9/20 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/current/working/directory'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_7 = '.py'
    var_8 = 0



# Parsed testcases at query #32
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_script.py'
    var_1 = module_0.run_script(var_0)
    assert var_1 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 3/6 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 3/6 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 3/7 statements.
# Partially parsed test_run_script_missing_shebang. Retrieved 3/7 statements.
# Partially parsed test_run_script_empty_file. Retrieved 4/7 statements.
# Partially parsed test_run_script_with_custom_cwd. Retrieved 3/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello")'
    var_2 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello"'
    var_2 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_fail_script.py'
    var_1 = 'import sys\nsys.exit(1)'
    var_2 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_no_shebang.sh'
    var_1 = 'echo "Hello"'
    var_2 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_empty.sh'
    var_1 = 'w'
    var_2 = open(var_0, var_1)
    var_3 = module_0.run_script(var_0)

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_script.py'
    var_2 = 'print("Hello")'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/some/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_run_script_with_context_extension_not_evaluated_to_false. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '/path/to/cwd'
    var_2 = {}
    var_3 = module_0.run_script_with_context(var_0, var_1, var_2)



# Parsed testcases at query #36
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #37
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_successful_hook. Retrieved 12/17 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_deletes_project. Retrieved 12/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_preserves_project. Retrieved 12/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = True
    var_10 = 'print("Success")'
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = True
    var_10 = 'import sys; sys.exit(1)'
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = True
    var_10 = 'import sys; sys.exit(1)'
    var_11 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #39
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_pre_prompt_script. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'hooks'
    var_2 = True
    var_3 = 'pre_prompt.py'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_pre_prompt_script. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('pre_prompt hook')"



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_on_failure. Retrieved 10/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_on_failure. Retrieved 6/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_extension. Retrieved 6/8 statements.


def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/current/working/directory'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.py'



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_work_in_changes_directory_and_restores.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/7 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/10 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre_gen.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'post_gen'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen'
    var_1 = 'nonexistent_directory'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen'
    var_1 = 'tests/test_hooks_multiple'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'pre_gen.py'
    var_5 = 'pre_gen.sh'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_oserror_errno_not_enoexec. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Permission denied'



# Parsed testcases at query #48
#--------------------------




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
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 6/8 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_successful_hook. Retrieved 10/11 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
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
    var_5 = 'test_project'
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
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #51
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #52
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/valid_script.py'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/valid_script.sh'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_script_successful_execution. Retrieved 2/10 statements.
# Partially parsed test_run_script_python_successful_execution. Retrieved 1/8 statements.
# Partially parsed test_run_script_failed_execution. Retrieved 2/11 statements.
# Partially parsed test_run_script_empty_file. Retrieved 1/9 statements.
# Partially parsed test_run_script_missing_shebang. Retrieved 2/11 statements.


def test_case_0():
    var_0 = "#!/bin/bash\necho 'Hello World'"
    var_1 = 493

def test_case_0():
    var_0 = "print('Hello World')"

def test_case_0():
    var_0 = '#!/bin/bash\nexit 1'
    var_1 = 493

def test_case_0():
    var_0 = 493

def test_case_0():
    var_0 = "echo 'Hello World'"
    var_1 = 493



# Parsed testcases at query #54
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #55
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #56
#--------------------------




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
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #57
#--------------------------




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
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_not_delete_project_on_failure. Retrieved 13/18 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = 'test_hook'
    var_2 = 'test_project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = None
    var_8 = lambda hook_name, project_dir, context: var_7
    var_9 = lambda repo_dir: var_7
    var_10 = var_8
    var_11 = var_9
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 6/9 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {var_0}
    var_2 = 'pre_gen_project'
    var_3 = 'tests/test_data/hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    var_5 = len(var_4)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {var_0}
    var_2 = 'invalid_hook'
    var_3 = 'tests/test_data/hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {var_0}
    var_2 = 'pre_gen_project'
    var_3 = 'nonexistent_directory'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {var_0}
    var_2 = 'pre_gen_project'
    var_3 = 'tests/test_data/hooks_with_backup'
    var_4 = module_0.find_hook(var_2, var_3)
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_5 == var_6

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = {var_0}
    var_2 = 'unsupported_hook'
    var_3 = 'tests/test_data/hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo_with_valid_hook'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo_with_invalid_hook'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo_with_no_hooks_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo_with_hook_failure'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo_with_multiple_hooks'
    var_1 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 3/10 statements.
# Partially parsed test_find_hook_with_invalid_hook. Retrieved 3/8 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 3/8 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'hook1.sh'
    var_1 = ''
    var_2 = 'hook1'

def test_case_0():
    var_0 = 'invalid_hook.sh'
    var_1 = ''
    var_2 = 'hook1'

def test_case_0():
    var_0 = 'hook1.sh~'
    var_1 = ''
    var_2 = 'hook1'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hook1'
    var_1 = 'non_existing_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'hook1.sh'
    var_1 = 'hook2.sh'
    var_2 = ''
    var_3 = ''
    var_4 = 'hook1'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_hook_existing_hook. Retrieved 6/14 statements.
# Partially parsed test_find_hook_non_existing_hook. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit'
    var_1 = 'hooks'
    var_2 = True
    var_3 = 'pre_commit.py'
    var_4 = ''
    var_5 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existing_hook'
    var_1 = 'hooks'
    var_2 = True
    var_3 = module_0.find_hook(var_0, var_1)
    assert var_3 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit'
    var_1 = 'non_existing_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_run_hook_with_valid_script. Retrieved 9/13 statements.
# Partially parsed test_run_hook_with_invalid_script_extension. Retrieved 9/13 statements.
# Partially parsed test_run_hook_with_backup_file. Retrieved 9/13 statements.
# Partially parsed test_run_hook_with_unsupported_hook. Retrieved 9/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project'
    var_4 = '.'
    var_5 = module_0.run_hook(var_3, var_4, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '#!/bin/sh\necho "test"'
    var_4 = 'hooks'
    var_5 = True
    var_6 = 'pre_gen_project'
    var_7 = '.'
    var_8 = module_0.run_hook(var_6, var_7, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = 'hooks'
    var_5 = True
    var_6 = 'pre_gen_project'
    var_7 = '.'
    var_8 = module_0.run_hook(var_6, var_7, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '#!/bin/sh\necho "test"'
    var_4 = 'hooks'
    var_5 = True
    var_6 = 'pre_gen_project'
    var_7 = '.'
    var_8 = module_0.run_hook(var_6, var_7, var_2)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '#!/bin/sh\necho "test"'
    var_4 = 'hooks'
    var_5 = True
    var_6 = 'invalid_hook'
    var_7 = '.'
    var_8 = module_0.run_hook(var_6, var_7, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 9/26 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '/path/to/cwd'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'rendered_content'
    var_8 = module_0.run_script_with_context(var_0, var_1, var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/15 statements.
# Partially parsed test_find_hook_with_invalid_hook_name. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_unsupported_hook. Retrieved 4/13 statements.
# Partially parsed test_find_hook_with_nonexistent_hooks_dir. Retrieved 2/7 statements.
# Partially parsed test_find_hook_with_multiple_valid_hooks. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = ''
    var_3 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook.py'
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
    var_0 = 'nonexistent_hooks'
    var_1 = 'pre_gen_project'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project.sh'
    var_3 = ''
    var_4 = ''
    var_5 = 'pre_gen_project'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_run_pre_prompt_hook_without_scripts. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/path/to/repo'



# Parsed testcases at query #8
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unsupported.py'
    var_1 = 'unsupported'
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
    var_0 = '/path/to/hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/.py'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #9
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'invalid_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = len(var_2)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/unsupported.py'
    var_1 = 'unsupported'
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
    var_0 = '/path/to/hook.py~'
    var_1 = 'hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/.py'
    var_1 = ''
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_run_script_success_python_script. Retrieved 1/8 statements.
# Partially parsed test_run_script_success_non_python_script. Retrieved 1/8 statements.
# Partially parsed test_run_script_failure_exit_status. Retrieved 1/9 statements.
# Failed to parse test_run_script_failure_empty_file.
# Partially parsed test_run_script_failure_invalid_path. Retrieved 1/6 statements.


def test_case_0():
    var_0 = "print('Hello, World!')"

def test_case_0():
    var_0 = "#!/bin/sh\necho 'Hello, World!'"

def test_case_0():
    var_0 = 'import sys\nsys.exit(1)'

def test_case_0():
    var_0 = 'nonexistent_script'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_run_hook_with_valid_script. Retrieved 6/14 statements.
# Partially parsed test_run_hook_with_invalid_script_extension. Retrieved 6/14 statements.
# Partially parsed test_run_hook_with_backup_file. Retrieved 6/14 statements.
# Partially parsed test_run_hook_with_unsupported_hook_name. Retrieved 6/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project'
    var_4 = '.'
    var_5 = module_0.run_hook(var_3, var_4, var_2)

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = b'#!/usr/bin/env python\nprint("Hello")'
    var_4 = 493
    var_5 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = b'#!/usr/bin/env python\nprint("Hello")'
    var_4 = 493
    var_5 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = b'#!/usr/bin/env python\nprint("Hello")'
    var_4 = 493
    var_5 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = b'#!/usr/bin/env python\nprint("Hello")'
    var_4 = 493
    var_5 = 'unsupported_hook'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_run_script_with_python_file_sets_run_thru_shell_true_on_windows. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'win'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_hook_valid_hook. Retrieved 4/7 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'valid_hook.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'backup_hook'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported_hook'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook'
    var_1 = 'tests/fixtures/non_existent'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 7/14 statements.
# Partially parsed test_run_pre_prompt_hook_with_no_hooks_dir. Retrieved 4/5 statements.
# Partially parsed test_run_pre_prompt_hook_with_no_pre_prompt_hook. Retrieved 5/8 statements.
# Partially parsed test_run_pre_prompt_hook_with_failed_hook. Retrieved 7/14 statements.


import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = True
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'print("Running pre_prompt hook")'
    var_5 = module_0.run_pre_prompt_hook(var_0)
    var_6 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = True
    var_2 = module_0.run_pre_prompt_hook(var_0)
    var_3 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = True
    var_2 = 'hooks'
    var_3 = module_0.run_pre_prompt_hook(var_0)
    var_4 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = True
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'exit(1)'
    var_5 = module_0.run_pre_prompt_hook(var_0)
    var_6 = module_1.rmtree(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_run_hook_with_no_scripts. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp'
    var_3 = {}



# Parsed testcases at query #17
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hooks_dir'
    var_1 = 'test_hook'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_run_hook_with_no_scripts. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = '/fake/dir'
    var_2 = 'pre_gen_project'



# Parsed testcases at query #19
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non-existing-hook'
    var_1 = 'tests/fixtures/hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre-commit'
    var_1 = 'non-existing-dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_script. Retrieved 3/16 statements.
# Partially parsed test_run_pre_prompt_hook_with_no_scripts. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'valid_script.py'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'test_repo'



# Parsed testcases at query #21
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_hook'
    var_1 = 'empty_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #22
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'some_hook'
    var_1 = 'some_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = 'script1.py'
    var_1 = 'script2.py'
    var_2 = [var_0, var_1]
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 == var_4
    assert var_5 is False



# Parsed testcases at query #24
#--------------------------




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
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 6/15 statements.
# Partially parsed test_find_hook_with_invalid_hook. Retrieved 6/12 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 6/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_0, var_1)
    var_5 = 'pre_gen_project.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'hooks'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_0, var_1)
    assert var_4 is None
    var_5 = 'pre_gen_project.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_0, var_1)
    assert var_4 is None
    var_5 = 'pre_gen_project.py~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'invalid_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_run_script_successful_python_script. Retrieved 1/9 statements.
# Partially parsed test_run_script_successful_non_python_script. Retrieved 2/11 statements.
# Partially parsed test_run_script_failed_exit_status. Retrieved 1/9 statements.
# Failed to parse test_run_script_failed_enoexec.
# Partially parsed test_run_script_failed_oserror. Retrieved 1/7 statements.


def test_case_0():
    var_0 = "print('Hello, World!')"

def test_case_0():
    var_0 = "#!/bin/sh\necho 'Hello, World!'"
    var_1 = 493

def test_case_0():
    var_0 = 'import sys\nsys.exit(1)'

def test_case_0():
    var_0 = 'nonexistent_script'



# Parsed testcases at query #27
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #28
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook.py'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook.py'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = 'unsupported_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook.py~'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook.py'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'valid_hook.py~'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook.py'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported_hook.py'
    var_1 = 'invalid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook.py~'
    var_1 = 'valid_hook'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_hook_with_valid_hook. Retrieved 4/7 statements.
# Partially parsed test_find_hook_with_backup_file. Retrieved 4/6 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre_gen_project.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre_gen_project.py~'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'unsupported_hook'
    var_1 = 'tests/test_hooks'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_find_hook_returns_list_of_valid_hooks. Retrieved 7/12 statements.
# Partially parsed test_find_hook_returns_none_for_invalid_hook. Retrieved 5/8 statements.
# Partially parsed test_find_hook_returns_none_for_empty_hooks_dir. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit'
    var_1 = 'hooks'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_0, var_1)
    var_5 = len(var_4)
    var_6 = 'pre_commit.py'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'invalid_hook'
    var_1 = 'hooks'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_0, var_1)
    assert var_4 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_commit'
    var_1 = 'hooks'
    var_2 = True
    var_3 = module_0.find_hook(var_0, var_1)
    assert var_3 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_valid_hook_returns_true_when_all_conditions_met. Retrieved 15/19 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/valid_hook.py'
    var_1 = 'valid_hook'
    var_2 = 'valid_hook'
    var_3 = [var_2]
    var_4 = ''
    var_5 = ()
    var_6 = {}
    var_7 = type(var_4, var_5, var_6)
    var_8 = ()
    var_9 = {}
    var_10 = type(var_4, var_8, var_9)
    var_11 = 'valid_hook.py'
    var_12 = '.py'
    var_13 = (var_2, var_12)
    var_14 = module_0.valid_hook(var_0, var_1)
    assert var_14 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_successful_hook. Retrieved 13/21 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_with_deletion. Retrieved 14/22 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_without_deletion. Retrieved 14/22 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Hook executed successfully")'
    var_11 = 'pre_gen_project'
    var_12 = module_0.run_hook_from_repo_dir(var_0, var_11, var_1, var_6, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'pre_gen_project'
    var_12 = True
    var_13 = module_0.run_hook_from_repo_dir(var_0, var_11, var_1, var_6, var_12)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = '/tmp/project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'pre_gen_project'
    var_12 = False
    var_13 = module_0.run_hook_from_repo_dir(var_0, var_11, var_1, var_6, var_12)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 3/12 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 3/11 statements.
# Failed to parse test_run_pre_prompt_hook_without_hooks_dir.
# Partially parsed test_run_pre_prompt_hook_with_failing_hook. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Hello, World!")'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'invalid_hook.py'
    var_2 = 'print("Hello, World!")'

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'import sys\nsys.exit(1)'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/script'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #35
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-pre-prompt'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-empty-hooks'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-no-hooks'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-backup-file'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-unsupported-hook'
    var_1 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_run_script_with_context. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'Test Project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = "\n    project_name = '{{ cookiecutter.project_name }}'\n    "
    var_6 = 'Rendered script should exist'



# Parsed testcases at query #37
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_successful_hook. Retrieved 13/19 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_with_delete. Retrieved 12/18 statements.
# Partially parsed test_run_hook_from_repo_dir_failed_hook_without_delete. Retrieved 13/20 statements.


import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

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
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_11 = module_1.rmtree(var_0)
    var_12 = module_1.rmtree(var_2)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

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
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_11 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

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
    var_9 = True
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_11 = module_1.rmtree(var_0)
    var_12 = module_1.rmtree(var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_does_not_delete_project_when_failed_hook_and_delete_project_on_failure_is_false. Retrieved 10/17 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test Project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #40
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script_fail.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script_invalid.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_run_script_with_non_py_script_and_non_executable. Retrieved 4/5 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'tests/data/non_executable_script.sh'
    var_1 = 'tests/data'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = 1



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = 0



# Parsed testcases at query #43
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #44
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #45
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'non_existent_script'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_success. Retrieved 10/12 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_with_delete. Retrieved 11/14 statements.
# Partially parsed test_run_hook_from_repo_dir_failure_without_delete. Retrieved 11/14 statements.
# Partially parsed test_run_hook_from_repo_dir_undefined_error. Retrieved 11/14 statements.


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
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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
    var_10 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

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



# Parsed testcases at query #47
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.sh'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/failing_script.py'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent_script.py'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/invalid_script.py'
    var_1 = '/path/to/cwd'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #48
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo/dir'
    var_1 = 'fake_hook'
    var_2 = '/fake/project/dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #49
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_temp_file_has_same_extension_as_original. Retrieved 7/12 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/script.py'
    var_1 = '/path/to/working/dir'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #51
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/non/existent/path/script.sh'
    var_1 = '/tmp'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deletes_project_on_failure. Retrieved 6/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/fake/project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_delete_project_on_failure. Retrieved 6/13 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = True
    var_5 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_run_pre_prompt_hook_with_valid_hook. Retrieved 5/10 statements.
# Partially parsed test_run_pre_prompt_hook_with_invalid_hook. Retrieved 5/9 statements.
# Partially parsed test_run_pre_prompt_hook_with_no_hooks_dir. Retrieved 3/4 statements.


import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = f'{var_0}/hooks'
    var_2 = 'print("pre_prompt hook executed")'
    var_3 = module_0.run_pre_prompt_hook(var_0)
    var_4 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = f'{var_0}/hooks'
    var_2 = 'print("invalid hook executed")'
    var_3 = module_0.run_pre_prompt_hook(var_0)
    var_4 = module_1.rmtree(var_0)

import cookiecutter.hooks as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = module_1.rmtree(var_0)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_run_script_with_context_creates_temp_file_with_correct_extension. Retrieved 8/14 statements.


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '/path/to/cwd'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_7 = '.py'



