####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles exceptions and directory deletion.'
    var_1 = '/fake/repo'
    var_2 = '/fake/project'
    var_3 = 'post_gen_project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when the hook runs successfully.'
    var_1 = '/fake/repo'
    var_2 = '/fake/project'
    var_3 = 'post_gen_project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook correctly identifies and executes hooks.'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Test run_hook_from_repo_dir handles success and failure scenarios,\n    specifically checking if project directory is deleted on failure.\n    '
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = 'Hook failed'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir success path.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_6, var_2, var_5, var_7)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the find_hook function with various scenarios.'
    var_1 = 'no_hooks'
    var_2 = 'pre_gen_project'
    var_3 = 'empty_hooks'
    var_4 = 'valid_hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = "#!/bin/bash\necho 'hello'"
    var_7 = 'unknown_hook.py'
    var_8 = 'post_gen_project.py~'
    var_9 = 'post_gen_project.py'
    var_10 = 'post_gen_project'
    var_11 = 'pre_prompt'
    var_12 = '~'
    var_13 = 0



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles success and failure scenarios.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'Hook failed'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir success scenario.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_6, var_2, var_5, var_7)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ project_name }}'
    var_6 = 'Hello test_project'
    var_7 = '.py'
    var_8 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_9 = 'utf-8'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook finds and executes scripts correctly.'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter_project_slug'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = module_0.run_hook(var_1, var_2, var_5)
    var_7 = '/tmp/hooks/pre_gen_project.py'
    var_8 = module_0.run_hook(var_1, var_2, var_5)
    var_9 = '/tmp/hooks/pre_gen_project.sh'
    var_10 = module_0.run_hook(var_1, var_2, var_5)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook propagates exceptions from the execution step.'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = '/tmp/hooks/post_gen_project.py'
    var_5 = 'Script failed'
    var_6 = module_0.run_hook(var_1, var_2, var_3)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook functionality.'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_0.run_pre_prompt_hook(var_0)
    var_6 = str(var_0)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook_from_repo_dir for various execution scenarios.\n    '
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'post_gen_project'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'Failed'
    var_8 = 'Undefined'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/original_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/original_dir'
    var_1 = '/path/to/pre_prompt.py'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = 'Script failed'
    var_5 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook finds and executes scripts correctly.'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = '/absolute/path/to/post_gen_project.py'
    var_7 = [var_6]
    var_8 = module_0.run_hook(var_1, var_2, var_5)
    var_9 = module_0.run_hook(var_1, var_2, var_5)
    var_10 = 0
    var_11 = var_7[var_10]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook executes all found scripts.'
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = '/path/1.py'
    var_5 = '/path/2.py'
    var_6 = [var_4, var_5]
    var_7 = module_0.run_hook(var_1, var_2, var_3)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'hooks_dir'
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None
    var_6 = 'pre_prompt'
    var_7 = 'hooks_dir'
    var_8 = module_0.find_hook(var_6, var_7)
    assert var_8 is None
    var_9 = 'pre_prompt'
    var_10 = 'hooks_dir'
    var_11 = module_0.find_hook(var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'pre_prompt'
    var_14 = 'hooks_dir'
    var_15 = module_0.find_hook(var_13, var_14)
    assert var_15 is None
    var_16 = 'pre_prompt'
    var_17 = 'hooks_dir'
    var_18 = module_0.find_hook(var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_script_with_context by verifying:\n    1. The Jinja2 rendering occurs correctly.\n    2. A temporary file is created with the rendered content.\n    3. run_script is called with the path to the temp file.\n    '
    assert var_0 == 'Hello world'
    var_1 = 0
    var_2 = '.py'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders template and calls run_script.'
    var_1 = 'Hello {{ name }}!'
    var_2 = 'post_gen_project.py'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context behavior when file reading fails.'
    var_1 = 'missing.py'
    var_2 = {}



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('hello')"
    var_3 = 'utf-8'
    var_4 = 'new_tmp_repo'
    var_5 = [var_4]
    var_6 = [var_1]
    var_7 = module_0.run_pre_prompt_hook(var_3)
    var_8 = 'pre_prompt_fail.py'
    var_9 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_10 = module_0.run_pre_prompt_hook(var_4)
    var_11 = str(var_4)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir handles hook failures and deletion correctly.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'post_gen_project'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that run_hook_from_repo_dir completes successfully when no error occurs.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'post_gen_project'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_hook_from_repo_dir for correct execution, \n    error handling, and directory cleanup.\n    '
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'post_gen_project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = 'Test Error'
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_8)
    var_10 = True
    var_11 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_10)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Tests run_hook_from_repo_dir when the hook executes successfully.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'post_gen_project'
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '/tmp/hooks/post_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = "print('Hello {{ project_name }}')"
    var_6 = "print('Hello test_project')"
    var_7 = 'utf-8'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'test_hook.py'
    var_1 = "#!/usr/bin/env python\nprint('hello')"
    var_2 = 'test_hook.sh'
    var_3 = "#!/bin/bash\necho 'hello'"
    var_4 = 'Exec format error'
    var_5 = 'Permission denied'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Tests run_hook_from_repo_dir for success, failure with cleanup, and failure without cleanup.'
    var_1 = '/tmp/repo'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '/tmp/hooks/post_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = "print('Hello {{ project_name }}')"
    var_6 = "print('Hello test_project')"
    var_7 = 'utf-8'



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook correctly identifies and executes discovered hooks.'
    var_1 = '/tmp/project'
    var_2 = var_0.args



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ project_name }}'
    var_6 = 'Hello test_project'
    var_7 = '.py'
    var_8 = 'utf-8'
    var_9 = '/tmp/tmp_script.py'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test successful execution of a hook.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test hook failure triggers rmtree if delete_project_on_failure is True.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_6, var_2, var_5, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test hook failure does NOT trigger rmtree if delete_project_on_failure is False.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_6, var_2, var_5, var_7)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with various scenarios.'
    var_1 = '/fake/repo'
    var_2 = '/fake/tmp_repo'
    var_3 = module_0.run_pre_prompt_hook(var_1)
    var_4 = '/fake/repo/hooks/pre_prompt.py'
    var_5 = [var_4]
    var_6 = '/fake/tmp_repo/hooks/pre_prompt.py'
    var_7 = [var_6]
    var_8 = module_0.run_pre_prompt_hook(var_1)
    var_9 = [var_4]
    var_10 = [var_6]
    var_11 = 'Script failed'
    var_12 = module_0.run_pre_prompt_hook(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test scenario where find_hook returns an empty list in the temp directory.'
    var_1 = '/fake/repo'
    var_2 = '/fake/tmp_repo'
    var_3 = '/fake/repo/hooks/pre_prompt.py'
    var_4 = [var_3]
    var_5 = []
    var_6 = module_0.run_pre_prompt_hook(var_1)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/pre_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ project_name }}'
    var_6 = 'Hello test_project'
    var_7 = '.py'
    var_8 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_9 = 'utf-8'
    var_10 = '/tmp/tmp_script.py'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Test that run_script_with_context correctly:\n    1. Reads the script file.\n    2. Renders it using Jinja2 with the provided context.\n    3. Writes the rendered content to a temporary file.\n    4. Calls run_script with the path to the temporary file and the target cwd.\n    '
    var_1 = '/fake/project/dir'
    var_2 = 'Hello test_project from tester!'
    var_3 = 'utf-8'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test that run_hook executes successfully without calling rmtree.'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'context'

def test_case_0():
    var_0 = 'Test that rmtree is called when a FailedHookException occurs.'
    var_1 = 'Hook failed'
    var_2 = 'project_dir'

def test_case_0():
    var_0 = 'Test that rmtree is called when an UndefinedError occurs.'
    var_1 = 'Undefined variable'
    var_2 = 'project_dir'

def test_case_0():
    var_0 = 'Test that rmtree is NOT called if delete_project_on_failure is False.'
    var_1 = 'Hook failed'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Test run_hook logic for finding and executing hooks.\n    '



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook with various scenarios: no hooks found, and executing multiple hooks.'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = module_0.run_hook(var_5, var_1, var_4)
    var_7 = '/tmp/hooks/pre_gen_project.py'
    var_8 = module_0.run_hook(var_5, var_1, var_4)
    var_9 = '/tmp/hooks/post_gen_project.py'
    var_10 = '/tmp/hooks/other_hook.py'
    var_11 = [var_9, var_10]
    var_12 = 'post_gen_project'
    var_13 = module_0.run_hook(var_12, var_1, var_4)
    var_14 = 0
    var_15 = var_11[var_14]
    var_16 = 1
    var_17 = var_11[var_16]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that exceptions in run_script_with_context propagate through run_hook.'
    var_1 = '/tmp/hooks/pre_gen_project.py'
    var_2 = 'Script failed'
    var_3 = 'pre_gen_project'
    var_4 = '/tmp/project'
    var_5 = {}
    var_6 = module_0.run_hook(var_3, var_4, var_5)
    var_7 = str(var_6)
    assert var_7 == 'Script failed'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes scripts when found and does nothing when not.'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'post_gen_project'
    var_6 = module_0.run_hook(var_5, var_1, var_4)
    var_7 = '/tmp/hooks/post_gen_project.py'
    var_8 = module_0.run_hook(var_5, var_1, var_4)
    var_9 = '/tmp/hooks/post_gen_project.sh'
    var_10 = module_0.run_hook(var_5, var_1, var_4)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that exceptions in run_script_with_context propagate through run_hook.'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'post_gen_project'
    var_6 = '/tmp/hooks/post_gen_project.py'
    var_7 = 'Script failed'
    var_8 = module_0.run_hook(var_5, var_1, var_4)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes the correct command and handles outcomes.'
    var_1 = '/tmp/test'
    var_2 = '/tmp/test'
    var_3 = OSError()



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook to ensure it correctly finds hooks and calls \n    run_script_with_context for each found script.\n    '
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = None
    var_4 = [var_3]
    var_5 = []



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook executes scripts found by find_hook.'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.py'
    var_3 = "#!/usr/bin/env python\nprint('hello')"
    var_4 = 'project'
    var_5 = 'name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = str(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook does nothing if no scripts are found.'
    var_1 = 'project'
    var_2 = {}
    var_3 = 'non_existent_hook'
    var_4 = module_0.run_hook(var_3, var_0, var_2)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '/tmp/hook.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = "print('{{ project_name }}')"
    var_6 = "print('test_project')"
    var_7 = 'utf-8'
    var_8 = '/tmp/temp_script.py'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/tmp/hooks/post_gen_project.py'
    var_1 = '/tmp/project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ project_name }}'
    var_6 = 'Hello test_project'
    var_7 = '.py'
    var_8 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_9 = 'utf-8'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test find_hook with various scenarios including missing dir, no hooks, and valid hooks.'
    var_1 = 'non_existent'
    var_2 = 'pre_gen_project'
    var_3 = 'empty_hooks'
    var_4 = 'invalid_hooks'
    var_5 = 'wrong_name.py'
    var_6 = 'pre_gen_project.py~'
    var_7 = 'not_a_hook.sh'
    var_8 = 'valid_hooks'
    var_9 = 'pre_prompt.py'
    var_10 = 'pre_gen_project.sh'
    var_11 = 'post_gen_project.py'
    var_12 = 'unrelated_script.py'
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = 'pre_prompt'
    var_15 = 0
    var_16 = 'post_gen_project'
    var_17 = 'non_existent_hook'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook functionality for various scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'hooks'
    var_5 = 'pre_gen_project'
    var_6 = module_0.find_hook(var_5, var_4)
    assert var_6 is None
    var_7 = 'pre_gen_project'
    var_8 = module_0.find_hook(var_7, var_4)
    assert var_8 is None
    var_9 = 'pre_gen_project'
    var_10 = 'pre_gen_project.py'
    var_11 = module_0.find_hook(var_9, var_7)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'pre_prompt'
    var_14 = module_0.find_hook(var_13, var_12)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'pre_prompt.py'
    var_17 = 'pre_prompt.sh'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Integration test using real filesystem.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = True
    var_4 = 'pre_gen_project.py'
    var_5 = "#!/bin/bash\necho 'hello'"
    var_6 = 'post_gen_project.py~'
    var_7 = "echo 'bad'"
    var_8 = 'pre_prompt.sh'
    var_9 = "#!/bin/bash\necho 'hi'"
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_prompt'
    var_16 = module_0.find_hook(var_15, var_11)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 'pre_prompt.sh'
    var_19 = 'non_existent'
    var_20 = module_0.find_hook(var_19, var_11)
    assert var_20 is None



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with success and failure scenarios.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'Failed'
    var_8 = True
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_8)
    var_10 = False
    var_11 = True
    assert var_11 is True
    assert var_11 is False
    var_12 = 'project_id'
    var_13 = locals()
    var_14 = var_12 in var_13

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that UndefinedError is also caught and triggers deletion if configured.'
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = 'Error'
    var_5 = 'pre_gen_project'
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_1, var_5, var_2, var_3, var_6)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Test run_hook:\n    1. Case where no scripts are found (should return early).\n    2. Case where scripts are found (should execute each script with context).\n    '
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = '/abs/path/to/post_gen_project.py'
    var_7 = '/abs/path/to/post_gen_project.sh'
    var_8 = [var_6, var_7]
    var_9 = module_0.run_hook(var_1, var_2, var_5)
    var_10 = module_0.run_hook(var_1, var_2, var_5)
    var_11 = 0
    var_12 = var_8[var_11]
    var_13 = 1
    var_14 = var_8[var_13]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Test that run_hook propagates exceptions raised during script execution.\n    '
    var_1 = 'post_gen_project'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = '/abs/path/to/error_script.py'
    var_5 = [var_4]
    var_6 = 'Script failed'
    var_7 = module_0.run_hook(var_1, var_2, var_3)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Wrapper for pytest compatibility. \n    The actual logic is contained in the TestRunHook class above.\n    '



# Parsed testcases at query #19
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_pre_prompt_hook for three scenarios:\n    1. No hook exists (returns original repo_dir).\n    2. Hook exists and runs successfully.\n    3. Hook exists but fails (raises FailedHookException).\n    '
    var_1 = 'empty_repo'
    var_2 = True
    var_3 = module_0.run_pre_prompt_hook(var_0)
    var_4 = 'hooks'
    var_5 = 'pre_prompt.py'
    var_6 = str(var_2)
    var_7 = 'Hook failed'
    var_8 = module_0.run_pre_prompt_hook(var_7)

def test_case_0():
    var_0 = 'Tests that it returns repo_dir if hooks directory is missing.'
    var_1 = 'no_hooks_folder'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = "#!/usr/bin/env python\nprint('hello')"
    var_4 = 'utf-8'
    var_5 = 'tmp_repo'
    var_6 = [var_1]
    var_7 = [var_2]
    var_8 = module_0.run_pre_prompt_hook(var_4)
    var_9 = [var_1]
    var_10 = [var_2]
    var_11 = 'Script failed'
    var_12 = module_0.run_pre_prompt_hook(var_1)



# Parsed testcases at query #21
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook with various scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks_dir'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_prompt'
    var_8 = 'hooks_dir'
    var_9 = module_0.find_hook(var_7, var_8)
    assert var_9 is None
    var_10 = 'pre_prompt'
    var_11 = 'hooks_dir'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'pre_prompt'
    var_15 = 'hooks_dir'
    var_16 = module_0.find_hook(var_14, var_15)
    assert var_16 is None
    var_17 = 'pre_prompt'
    var_18 = 'hooks_dir'
    var_19 = module_0.find_hook(var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 1



# Parsed testcases at query #22
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_pre_prompt_hook with various scenarios.'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'
    var_5 = 'new_tmp_repo'
    var_6 = module_0.run_pre_prompt_hook(var_0)
    var_7 = 'Script failed'
    var_8 = module_0.run_pre_prompt_hook(var_7)
    var_9 = 'hooks'
    var_10 = 'pre_prompt.py'
    var_11 = var_8 / var_10
    var_12 = "print('hello')"



# Parsed testcases at query #23
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo/dir'
    var_1 = '/fake/tmp/repo/dir'
    var_2 = module_0.run_pre_prompt_hook(var_0)
    var_3 = 'Hook failed'
    var_4 = module_0.run_pre_prompt_hook(var_0)
    var_5 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test behavior when find_hook returns an empty list explicitly.'
    var_1 = '/fake/repo/dir'
    var_2 = module_0.run_pre_prompt_hook(var_1)



