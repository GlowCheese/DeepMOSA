####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = "Test that run_script_with_context raises error if script path doesn't exist."
    var_1 = 'non_existent.py'
    var_2 = '.'
    var_3 = {}



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook with various directory and file scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks_dir'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_prompt.py'
    var_8 = 'pre_gen_project.sh'
    var_9 = 'post_gen_project.py'
    var_10 = 'pre_prompt.py~'
    var_11 = 'wrong_hook.py'
    var_12 = 'other_hook.py'
    var_13 = 'pre_prompt.txt'
    var_14 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13]
    var_15 = 'pre_prompt'
    var_16 = 'hooks_dir'
    var_17 = module_0.find_hook(var_15, var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'pre_prompt.py'
    var_20 = 'pre_prompt.txt'
    var_21 = 'post_gen_project'
    var_22 = module_0.find_hook(var_21, var_16)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'post_gen_project.py'
    var_25 = 'non_existent'
    var_26 = module_0.find_hook(var_25, var_16)
    assert var_26 is None



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '/tmp/repo'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '/tmp/repo'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Test run_pre_prompt_hook with various scenarios:\n    1. No pre_prompt hook exists.\n    2. pre_prompt hook exists and runs successfully.\n    3. pre_prompt hook exists but fails.\n    '
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('hello')"
    var_5 = 'utf-8'
    var_6 = 'new_tmp_dir'
    var_7 = module_0.run_pre_prompt_hook(var_0)
    var_8 = 'Hook failed'
    var_9 = str(var_2)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Tests the find_hook function with various scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks_dir'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_gen_project'
    var_8 = 'hooks_dir'
    var_9 = module_0.find_hook(var_7, var_8)
    assert var_9 is None
    var_10 = 'pre_gen_project'
    var_11 = 'hooks_dir'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = 'post_gen_project'
    var_14 = module_0.find_hook(var_13, var_11)
    var_15 = 'pre_gen_project'
    var_16 = 'hooks_dir'
    var_17 = module_0.find_hook(var_15, var_16)
    assert var_17 is None
    var_18 = 'pre_gen_project'
    var_19 = 'hooks_dir'
    var_20 = module_0.find_hook(var_18, var_19)
    assert var_20 is None



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script with various scenarios: success, python script, non-python script, and failure.'
    var_1 = 'test_script.py'
    var_2 = "import sys; print('hello'); sys.exit(0)"
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'test_script.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = module_0.run_script(var_0, var_1)
    var_7 = 'fail_script.py'
    var_8 = 'import sys; sys.exit(1)'
    var_9 = module_0.run_script(var_0, var_1)
    var_10 = 'empty.py'
    var_11 = ''
    var_12 = 'Exec format error'
    var_13 = module_0.run_script(var_0, var_12)
    var_14 = 'broken.py'
    var_15 = "print('fail')"
    var_16 = 'Permission denied'
    var_17 = module_0.run_script(var_0, var_16)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = '/tmp/project'
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}



# Parsed testcases at query #9
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests the run_script function for various scenarios:\n    1. Successful execution of a Python script.\n    2. Successful execution of a shell script.\n    3. Failure of a script (non-zero exit status).\n    4. Failure due to OSError (e.g., ENOEXEC).\n    '
    var_1 = 'test_script.py'
    var_2 = 'import sys; sys.exit(0)'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'test_hook.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = module_0.run_script(var_0, var_1)
    var_7 = 'fail_script.py'
    var_8 = 'import sys; sys.exit(1)'
    var_9 = module_0.run_script(var_0, var_1)
    var_10 = OSError()
    var_11 = module_0.run_script(var_0, var_1)
    var_12 = 'Permission denied'
    var_13 = OSError(var_0, var_12)
    var_14 = module_0.run_script(var_0, var_12)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the find_hook function with various scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks_dir'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_gen_project.py'
    var_8 = 'hooks_dir'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks_dir'
    var_11 = module_0.find_hook(var_9, var_10)
    var_12 = 'pre_gen_project.py'
    var_13 = 'post_gen_project'
    var_14 = 'pre_prompt'
    var_15 = 'hooks_dir'
    var_16 = module_0.find_hook(var_14, var_15)
    assert var_16 is None
    var_17 = 'invalid_hook'
    var_18 = 'hooks_dir'
    var_19 = module_0.find_hook(var_17, var_18)
    assert var_19 is None



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Test run_script_with_context:\n    1. Mocks create_env_with_context to return a mock Jinja2 environment.\n    2. Mocks the template rendering process.\n    3. Verifies that run_script is called with the path to the rendered temporary file.\n    '
    var_1 = 'post_gen_project.py'
    var_2 = "print('Hello {{ name }}')"
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = "print('Hello World')"
    var_7 = 'project_dir'
    var_8 = 'utf-8'
    var_9 = 'temp_script.py'
    var_10 = 'utf-8'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook by mocking find_hook and run_script_with_context.\n    Verifies that scripts are found and executed with the correct arguments.\n    '
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = var_0.args



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '/tmp'
    var_1 = 'win'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = module_0.run_script(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'script.sh'
    var_2 = module_0.run_script(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = OSError()
    var_1 = 'script.sh'
    var_2 = module_0.run_script(var_1)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '/path/to/hook.py'
    var_1 = '/path/to/project'
    var_2 = 'Hello {{ project_name }}!'
    var_3 = 'Hello test_project!'
    var_4 = 'utf-8'
    var_5 = '/tmp/temp_script.py'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook_from_repo_dir for correct execution, \n    exception handling, and conditional directory deletion.\n    '
    var_1 = '/fake/repo'
    var_2 = '/fake/project'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'Error occurred'
    var_8 = 'Unexpected error'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook with various scenarios including missing dir, no hooks, and valid hooks.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_gen_project'
    var_8 = 'pre_gen_project.py'
    var_9 = 'hooks'
    var_10 = 'hooks'
    var_11 = module_0.find_hook(var_7, var_10)
    var_12 = [var_9]
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = module_0.find_hook(var_13, var_14)
    assert var_15 is None
    var_16 = 'pre_gen_project'
    var_17 = 'hooks'
    var_18 = module_0.find_hook(var_16, var_17)
    assert var_18 is None



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test that it returns original dir if no pre_prompt hook exists.'

def test_case_0():
    var_0 = 'Test successful execution of a pre_prompt hook.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsysys.exit(0)'
    var_4 = 'new_tmp'
    var_5 = [var_4]
    var_6 = [var_2]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that FailedHookException is re-raised during pre_prompt execution.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_4 = 'new_tmp'
    var_5 = [var_4]
    var_6 = [var_2]
    var_7 = 'Script failed'
    var_8 = module_0.run_pre_prompt_hook(var_4)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the find_hook function with various scenarios.'
    var_1 = 'pre_gen_project'
    var_2 = 'non_existent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = module_0.find_hook(var_4, var_5)
    assert var_6 is None
    var_7 = 'pre_prompt'
    var_8 = 'hooks'
    var_9 = module_0.find_hook(var_7, var_8)
    assert var_9 is None
    var_10 = 'pre_prompt.py'
    var_11 = 'hooks'
    var_12 = 'pre_prompt'
    var_13 = 'hooks'
    var_14 = module_0.find_hook(var_12, var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 0
    var_17 = var_14[var_16]
    var_18 = 'pre_prompt'
    var_19 = 'hooks'
    var_20 = module_0.find_hook(var_18, var_19)
    assert var_20 is None
    var_21 = 'pre_prompt'
    var_22 = 'hooks'
    var_23 = module_0.find_hook(var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 1



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Test run_script_with_context:\n    1. Verifies that the script content is read.\n    2. Verifies that Jinja2 renders the content with the provided context.\n    3. Verifies that a temporary file is created with the rendered content.\n    4. Verifies that run_script is called with the temporary file path and cwd.\n    '
    var_1 = 'Hello {{ name }}!'
    var_2 = 'pre_gen_project.py'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = 'temp_script.py'
    var_8 = b'Hello World!'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook_from_repo_dir for correct handling of hook failures,\n    deletion of project directory, and re-raising of specific exceptions.\n    '
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_hook_from_repo_dir when the hook executes successfully.\n    '
    var_1 = '/tmp/repo'
    var_2 = '/tmp/project'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_7)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test that the function returns the original repo_dir if no pre_prompt hook exists.'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that the function executes the pre_prompt hook if it exists.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsys_exit = 0\n'
    var_4 = 'utf-8'
    var_5 = module_0.run_pre_prompt_hook(var_0)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test that FailedHookException is raised if the pre_prompt hook fails.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)\n'
    var_4 = 'utf-8'
    var_5 = 'Hook failed'
    var_6 = module_0.run_pre_prompt_hook(var_5)
    var_7 = str(var_5)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests the run_script function for various scenarios including \n    successful execution, Python script execution, and failure handling.\n    '
    var_1 = 'test_hook.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'test_hook.py'
    var_5 = 'import sys; sys.exit(0)'
    var_6 = module_0.run_script(var_0, var_1)
    var_7 = module_0.run_script(var_0, var_1)
    var_8 = OSError()
    var_9 = module_0.run_script(var_0, var_1)
    var_10 = OSError()
    var_11 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_pre_prompt_hook for various scenarios:\n    - No hooks present (returns original repo_dir)\n    - Hooks present and run successfully (returns new tmp_repo_dir)\n    - Hooks present but execution fails (raises FailedHookException)\n    '
    var_1 = '/original/repo_dir'
    var_2 = '/tmp/tmp_repo_dir'
    var_3 = None
    var_4 = 'Hook failed'
    var_5 = module_0.run_pre_prompt_hook(var_1)
    var_6 = module_0.run_pre_prompt_hook(var_1)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_pre_prompt_hook for various scenarios:\n    1. No hook found: Returns original repo_dir.\n    2. Hook found and succeeds: Returns new tmp_repo_dir.\n    3. Hook found but fails: Raises FailedHookException.\n    4. Hook name is invalid: Returns original repo_dir.\n    '
    var_1 = '/fake/repo_dir'
    var_2 = '/fake/tmp_repo_dir'
    var_3 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_4 = [var_3]
    var_5 = '/fake/tmp_repo_dir/hooks/pre_prompt.py'
    var_6 = [var_5]
    var_7 = 'Hook failed'
    var_8 = module_0.run_pre_prompt_hook(var_1)
    var_9 = module_0.run_pre_prompt_hook(var_1)
    var_10 = module_0.run_pre_prompt_hook(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Specific test to ensure the exception message is wrapped correctly.'
    var_1 = '/fake/repo_dir'
    var_2 = '/fake/tmp_repo_dir'
    var_3 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_4 = [var_3]
    var_5 = '/fake/tmp_repo_dir/hooks/pre_prompt.py'
    var_6 = [var_5]
    var_7 = 'Original Error'
    var_8 = module_0.run_pre_prompt_hook(var_1)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests run_hook logic for both the case where no hooks are found\n    and the case where multiple hooks are found and executed.\n    '
    var_1 = '/tmp/project'

def test_case_0():
    var_0 = '\n    A more detailed test simulating a successful execution flow \n    of run_hook with a single valid script.\n    '
    var_1 = 'post_gen_project'
    var_2 = '/tmp/output'
    var_3 = 'user'
    var_4 = 'tester'
    var_5 = {var_3: var_4}
    var_6 = '/tmp/template/hooks/post_gen_project.py'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '/tmp/repo'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Test that run_script_with_context correctly renders a Jinja2 template,\n    writes it to a temporary file, and calls run_script.\n    '
    var_1 = 'Hello tester! Welcome to test_project.'
    var_2 = 0

def test_case_0():
    var_0 = '\n    Test that run_script_with_context preserves the file extension in the temp file.\n    '
    var_1 = 'hook.sh'
    var_2 = "#!/bin/bash\necho 'test'"
    var_3 = 'utf-8'
    var_4 = 0
    var_5 = '.sh'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Test run_hook execution logic.\n    Verifies that find_hook is called and run_script_with_context \n    is called the correct number of times.\n    '
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Test that run_hook returns early if no scripts are found.\n    '
    var_1 = 'non_existent_hook'
    var_2 = '/tmp/project'
    var_3 = {}
    var_4 = module_0.run_hook(var_1, var_2, var_3)



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Tests run_pre_prompt_hook with various scenarios:\n    1. No pre_prompt hook exists (returns original repo_dir).\n    2. pre_prompt hook exists and runs successfully.\n    3. pre_prompt hook exists but fails (raises FailedHookException).\n    '
    var_1 = '/fake/repo/dir'
    var_2 = '/fake/repo/dir/hooks/pre_prompt.py'
    var_3 = [var_2]
    var_4 = '/tmp/tmp_repo/hooks/pre_prompt.py'
    var_5 = [var_4]
    var_6 = 'Hook failed'
    var_7 = '/tmp/tmp_repo'
    var_8 = module_0.run_pre_prompt_hook(var_1)
    var_9 = module_0.run_pre_prompt_hook(var_1)
    var_10 = module_0.run_pre_prompt_hook(var_1)
    var_11 = str(var_10)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the run_script function with various scenarios.'
    var_1 = 'test_hook.py'
    var_2 = 'import sys; sys.exit(0)'
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = 'test_hook.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = module_0.run_script(var_0, var_1)
    var_7 = module_0.run_script(var_0, var_1)
    var_8 = 'No such file or directory'
    var_9 = module_0.run_script(var_0, var_8)
    var_10 = 'Exec format error'
    var_11 = module_0.run_script(var_0, var_10)



