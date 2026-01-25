####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'project_name'
    var_3 = 'Test Project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'print("Hello, World!")'
    var_3 = 'pre_gen_project'
    var_4 = module_0.find_hook(var_3, var_0)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'pre_gen_project.py'
    var_9 = 'hooks'
    var_10 = 'print("Hello, World!")'
    var_11 = module_0.find_hook(var_3, var_9)
    assert var_11 is None
    var_12 = 'random_script.py'
    var_13 = 'invalid_hooks'
    var_14 = module_0.find_hook(var_3, var_13)
    assert var_14 is None
    var_15 = 'hooks'
    var_16 = 'print("Hello, World!")'
    var_17 = module_0.find_hook(var_3, var_15)
    assert var_17 is None
    var_18 = 'pre_gen_project.py~'
    var_19 = 'hooks'
    var_20 = 'print("Hello, World!")'
    var_21 = 'unsupported_hook'
    var_22 = module_0.find_hook(var_21, var_19)
    assert var_22 is None
    var_23 = 'unsupported_hook.py'
    var_24 = 'hooks'
    var_25 = 'print("Hello, World!")'
    var_26 = 'echo "Hello, World!"'
    var_27 = module_0.find_hook(var_3, var_24)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'pre_gen_project.sh'
    var_30 = 'hooks'
    var_31 = 'print("Hello, World!")'
    var_32 = 'print("Hello, World!")'
    var_33 = module_0.find_hook(var_3, var_30)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = var_33[var_6]
    var_36 = 'hooks'
    var_37 = 'print("Hello, World!")'
    var_38 = 'print("Hello, World!")'
    var_39 = module_0.find_hook(var_3, var_36)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = var_39[var_6]
    var_42 = 'hooks'
    var_43 = 'print("Hello, World!")'
    var_44 = 'print("Hello, World!")'
    var_45 = module_0.find_hook(var_3, var_42)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = var_45[var_6]
    var_48 = 'hooks'
    var_49 = 'print("Hello, World!")'
    var_50 = 'echo "Hello, World!"'
    var_51 = 'print("Hello, World!")'
    var_52 = module_0.find_hook(var_3, var_48)
    var_53 = len(var_52)
    assert var_53 == 2



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook_from_repo_dir function.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'print("Hello from hook")'
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'Test Project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'
    var_11 = True
    var_12 = 'exit(1)'
    var_13 = 'pre_gen_project'
    var_14 = True
    var_15 = 'exit(1)'
    var_16 = 'pre_gen_project'
    var_17 = False



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('Pre-prompt hook executed')"
    var_3 = 'invalid_pre_prompt.py'
    var_4 = ''



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = 'test'
    var_3 = {var_2: var_2}



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the find_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = True
    var_5 = module_0.find_hook(var_1)
    assert var_5 is None
    var_6 = 'print("Hello, World!")'
    var_7 = module_0.find_hook(var_1)
    var_8 = 'print("Hello, World!")'
    var_9 = module_0.find_hook(var_1)
    var_10 = 'hooks/pre_gen_project.py'
    var_11 = 'hooks/pre_gen_project.py~'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, world!")'
    var_2 = 'nonexistent.py'
    var_3 = 'empty_script.py'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt.py'
    var_1 = 'pre_prompt'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True
    var_3 = 'pre_prompt.sh'
    var_4 = module_0.valid_hook(var_3, var_1)
    assert var_4 is True
    var_5 = 'pre_prompt.py~'
    var_6 = module_0.valid_hook(var_5, var_1)
    assert var_6 is False
    var_7 = 'post_gen_project.py'
    var_8 = module_0.valid_hook(var_7, var_1)
    assert var_8 is False
    var_9 = 'invalid_hook.py'
    var_10 = module_0.valid_hook(var_9, var_1)
    assert var_10 is False



# Parsed testcases at query #10
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = '/path/to/repo'
    var_9 = 'pre_gen_project'
    var_10 = '/path/to/project'
    var_11 = {var_3: var_4}
    var_12 = True
    var_13 = module_0.run_hook_from_repo_dir(var_8, var_9, var_10, var_11, var_12)
    var_14 = '/path/to/repo'
    var_15 = 'pre_gen_project'
    var_16 = '/path/to/project'
    var_17 = {var_13: var_4}
    var_18 = False
    var_19 = module_0.run_hook_from_repo_dir(var_14, var_15, var_16, var_17, var_18)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'path/to/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = 'path/to/repo_with_pre_prompt_hook'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    var_4 = 'path/to/repo_with_failing_pre_prompt_hook'
    var_5 = module_0.run_pre_prompt_hook(var_4)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Pre-prompt hook executed")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'exit(1)'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/project'
    var_4 = 'pre_gen_project'
    var_5 = module_0.run_hook(var_4, var_3, var_2)
    var_6 = 'invalid_hook'
    var_7 = module_0.run_hook(var_6, var_3, var_2)
    var_8 = 'pre_gen_project'
    var_9 = module_0.run_hook(var_8, var_3, var_2)
    var_10 = 'post_gen_project'
    var_11 = module_0.run_hook(var_10, var_3, var_2)
    var_12 = 'post_gen_project'
    var_13 = module_0.run_hook(var_12, var_3, var_2)
    var_14 = 'post_gen_project'
    var_15 = module_0.run_hook(var_14, var_3, var_2)
    var_16 = 'All tests passed.'
    var_17 = print(var_16)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir function.'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.sh'
    var_6 = True
    var_7 = '#!/bin/bash\necho "Running pre_gen_project hook"'
    var_8 = 'pre_gen_project'
    var_9 = True
    var_10 = '#!/bin/bash\nexit 1'
    var_11 = 'pre_gen_project'
    var_12 = True



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the run_script_with_context function.'
    var_1 = 'test_script.py'
    var_2 = 'print("Hello, {{ cookiecutter.name }}!")'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'non_existent_script.py'
    var_9 = 'test_script.py'
    var_10 = 'raise ValueError("Error")'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the run_pre_prompt_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'print("Running pre_prompt hook")'
    var_4 = 'hooks'
    var_5 = 'pre_prompt.py'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = 'import sys; sys.exit(1)'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'fake_repo_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'fake_project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True



# Parsed testcases at query #18
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '#!/bin/bash\n'
    var_1 = 'exit 0\n'
    var_2 = '#!/bin/bash\n'
    var_3 = 'exit 1\n'
    var_4 = module_0.run_script(var_2)
    var_5 = ''
    var_6 = module_0.run_script(var_5)
    var_7 = '#!/bin/bash\n'
    var_8 = 'exit 0\n'
    var_9 = '/nonexistent/path/to/script.sh'
    var_10 = module_0.run_script(var_9)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'pre_prompt.py'
    var_1 = 'print("Pre-Prompt Hook Executed")'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test the run_script_with_context function.'
    var_1 = 'test_script.py'
    var_2 = 'print("Hello, {{ name }}!")'
    assert var_2 == 'print("Hello, World!")'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #21
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = '/path/to/script'
    var_2 = [var_1]
    var_3 = lambda hook_name, hooks_dir='hooks': var_2
    var_4 = None
    var_5 = lambda script_path, cwd, context: var_4
    var_6 = 'pre_gen_project'
    var_7 = '/project/dir'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = lambda hook_name, hooks_dir='hooks': var_4
    var_13 = {var_8: var_9}
    var_14 = module_0.run_hook(var_6, var_7, var_13)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py'
    var_7 = ''
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.py~'
    var_11 = ''
    var_12 = 'pre_gen_project'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.sh'
    var_2 = '#!/bin/bash\necho "Hello World"'
    var_3 = 'invalid_hook.sh'
    var_4 = '#!/bin/bash\necho "Invalid Hook"'
    var_5 = 'pre_gen_project'
    var_6 = 'invalid_hook'
    var_7 = 'post_gen_project'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = True
    var_3 = 'print("Pre-prompt hook executed")'
    var_4 = 'import sys; sys.exit(1)'



# Parsed testcases at query #25
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = '/path/to/repo_with_hook'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    var_4 = '/path/to/repo_with_failing_hook'
    var_5 = module_0.run_pre_prompt_hook(var_4)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = '#!/bin/sh\necho "Hello, World!"\n'
    var_1 = 493



# Parsed testcases at query #28
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo/dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = '/fake/repo/dir/hooks/pre_prompt.sh'
    var_3 = module_0.run_pre_prompt_hook(var_0)
    var_4 = module_0.run_pre_prompt_hook(var_0)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'nonexistent.py'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1, var_0)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = True
    var_5 = 'invalid_hook'
    var_6 = module_0.find_hook(var_5, var_3)
    assert var_6 is None
    var_7 = 'pre_gen_project.py'
    var_8 = ''
    var_9 = module_0.find_hook(var_8, var_3)



# Parsed testcases at query #31
#--------------------------


import cookiecutter.utils as module_0
import cookiecutter.hooks as module_1

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = "print('Hello, World!')"
    var_3 = module_0.make_executable(var_0)
    var_4 = module_1.run_script(var_0, var_1)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = "#!/bin/bash\n\necho 'Hello, World!'"
    var_1 = 493
    var_2 = '#!/bin/bash\n\nexit 1'
    var_3 = "#!/bin/bash\n\necho 'Hello, World!'"



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir function.'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'pre_gen_project'
    var_5 = True
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = True
    var_9 = 'print("Running pre_gen_project hook")'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = "echo '{{ greeting }}'"
    var_1 = 'greeting'
    var_2 = 'Hello, World!'
    var_3 = {var_1: var_2}



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('Pre-prompt hook executed')"
    var_3 = "raise Exception('Hook failed')"



# Parsed testcases at query #36
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'pre_gen_project.py'
    var_5 = 'pre_gen_project'
    var_6 = 'pre_gen_project.py'
    var_7 = 'post_gen_project.py'
    var_8 = 'pre_gen_project'
    var_9 = 'pre_gen_project.py~'
    var_10 = 'pre_gen_project'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Running pre_prompt hook")'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context function.'
    var_1 = 'echo {{ message }}'
    var_2 = 'message'
    var_3 = 'Hello, World!'
    var_4 = {var_2: var_3}



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_project'
    var_4 = '/tmp/hooks/pre_gen_project.py'
    var_5 = [var_4]
    var_6 = lambda hook_name, hooks_dir='hooks': var_5
    var_7 = None
    var_8 = lambda script_path, cwd, context: var_7
    var_9 = 'pre_gen_project'



# Parsed testcases at query #40
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = '/path/to/script'
    var_2 = [var_1]
    var_3 = lambda hook_name, hooks_dir='hooks': var_2
    var_4 = None
    var_5 = lambda script_path, cwd, context: var_4
    var_6 = 'pre_gen_project'
    var_7 = '.'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = lambda hook_name, hooks_dir='hooks': var_9
    var_13 = 'pre_gen_project'
    var_14 = '.'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = module_0.run_hook(var_13, var_14, var_17)



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho "Hello, World!"\n'
    var_2 = 493
    var_3 = 'nonexistent.sh'
    var_4 = 'error_script.sh'
    var_5 = '#!/bin/bash\nexit 1\n'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('Running pre_prompt hook')"
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'import sys; sys.exit(1)'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = True
    var_3 = "print('Pre-prompt hook executed')"



# Parsed testcases at query #44
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'test_dir'
    var_2 = module_0.run_script(var_0, var_1)
    var_3 = module_0.run_script(var_0, var_1)
    var_4 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #45
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'pre_gen_project.py'
    var_5 = 'print("Hello, World!")'
    var_6 = 'pre_gen_project'
    var_7 = 'pre_gen_project.py~'
    var_8 = 'print("Hello, World!")'
    var_9 = 'pre_gen_project'
    var_10 = 'unsupported_hook.py'
    var_11 = 'print("Hello, World!")'
    var_12 = 'pre_gen_project'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho "Hello, World!"'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'project_name'
    var_4 = {var_3: var_2}
    var_5 = True
    var_6 = True
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'print("Running pre_gen_project hook")'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'import sys; sys.exit(0)'
    var_3 = 'project'
    var_4 = {}
    var_5 = 'post_gen_project'
    var_6 = True



# Parsed testcases at query #49
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '\n    Test run_hook function.\n    '
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/test_project'
    var_3 = 'project_name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = module_0.run_hook(var_1, var_2, var_5)
    var_7 = module_0.run_hook(var_1, var_2, var_5)
    var_8 = module_0.run_hook(var_1, var_2, var_5)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'print("Hello, world!")'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py~'
    var_10 = 'print("Hello, world!")'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py'
    var_14 = 'pre_gen_project.sh'
    var_15 = 'print("Hello, world!")'
    var_16 = 'echo "Hello, world!"'
    var_17 = 'pre_gen_project'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '/tmp/test_script.py'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = "print('{{ project_name }}')"
    var_6 = 'utf-8'
    var_7 = None



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = 'test_hook'
    var_2 = 'test_project_dir'
    var_3 = 'test_key'
    var_4 = 'test_value'
    var_5 = {var_3: var_4}
    var_6 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('Running pre_prompt hook')"



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("pre_prompt hook executed")'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = "#!/bin/bash\necho 'Hello, World!'"



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = True
    var_3 = ''
    var_4 = module_0.find_hook(var_1, var_0)
    var_5 = 'pre_gen_project.py'
    var_6 = 'no_hooks'
    var_7 = 'pre_gen_project'
    var_8 = module_0.find_hook(var_7, var_6)
    assert var_8 is None
    var_9 = 'hooks'
    var_10 = 'invalid_hook'
    var_11 = module_0.find_hook(var_10, var_9)
    assert var_11 is None
    var_12 = 'hooks'
    var_13 = 'pre_gen_project'
    var_14 = ''
    var_15 = module_0.find_hook(var_13, var_12)
    assert var_15 is None
    var_16 = 'pre_gen_project.py~'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '/mock/repo_dir'
    var_1 = '/mock/project_dir'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'variable'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = "print('{{ variable }}')"



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello, {{ name }}!')"
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'mock_script.py'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/mock/cwd'
    var_5 = None
    var_6 = lambda path, cwd: var_5
    var_7 = 'mock_content'
    var_8 = lambda context: var_5
    var_9 = lambda content: var_5
    var_10 = 'mock_output'
    var_11 = lambda **kwargs: var_10
    var_12 = lambda content: var_5
    var_13 = module_0.run_script_with_context(var_0, var_4, var_3)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook_from_repo_dir function.'
    var_1 = 'tests'
    var_2 = 'fake-repo-pre'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = True
    var_10 = 'pre_gen_project'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = '/path/to/script.py'
    var_2 = [var_1]
    var_3 = lambda hook_name, hooks_dir='hooks': var_2
    var_4 = None
    var_5 = lambda script_path, cwd, context: var_4
    var_6 = 'pre_gen_project'
    var_7 = '.'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = lambda hook_name, hooks_dir='hooks': var_9
    var_13 = 'pre_gen_project'
    var_14 = '.'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = module_0.run_hook(var_13, var_14, var_17)
    var_19 = 'All tests passed for run_hook'
    var_20 = print(var_19)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = True
    var_4 = 'print("Hello from pre_gen_project hook")'
    var_5 = 'project'
    var_6 = 'pre_gen_project'
    var_7 = {}



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hello from hook!")'
    var_3 = 493
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Running pre_prompt hook")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'raise Exception("Failed hook")'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = True
    var_2 = 'pre_prompt.py'
    var_3 = 'import sys\nsys.exit(0)'
    var_4 = 'import sys\nsys.exit(1)'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context function.'
    var_1 = 'print("Hello, World!")'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context function.'
    var_1 = 'test_script.py'
    var_2 = 'print("Hello, {{ name }}!")'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #21
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'pre_gen_project.py'
    var_5 = 'pre_gen_project'
    var_6 = 'pre_gen_project.py~'
    var_7 = 'pre_gen_project'
    var_8 = 'unsupported_hook.py'
    var_9 = 'unsupported_hook'
    var_10 = 'pre_gen_project.py'
    var_11 = 'pre_gen_project.sh'
    var_12 = 'pre_gen_project'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '\nimport os\nprint("Template variable: {{ my_var }}")\n    '
    var_2 = 'my_var'
    var_3 = 'Hello, World!'
    var_4 = {var_2: var_3}



# Parsed testcases at query #23
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = module_0.find_hook(var_0)
    assert var_1 is None
    var_2 = 'pre_gen_project.py'
    var_3 = 'print("Hello World")'
    var_4 = 'invalid_hook.py'
    var_5 = 'print("Hello World")'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '{{ project_name }}'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = '.'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the run_pre_prompt_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = '#!/usr/bin/env python\nprint("Pre-prompt hook")'
    var_4 = 'nonexistent'
    var_5 = 'failing_pre_prompt.py'
    var_6 = '#!/usr/bin/env python\nraise Exception("Hook failed")'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir function.'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.sh'
    var_6 = '#!/bin/bash\necho "Running pre_gen_project hook"'
    var_7 = 'pre_gen_project'
    var_8 = True
    var_9 = '#!/bin/bash\nexit 1'
    var_10 = 'pre_gen_project'
    var_11 = True



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py'
    var_7 = 'print("Hello")'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.py'
    var_11 = 'pre_gen_project.sh'
    var_12 = 'print("Hello")'
    var_13 = 'echo "Hello"'
    var_14 = 'pre_gen_project'
    assert var_14 is None
    var_15 = 'hooks'
    var_16 = 'pre_gen_project.py~'
    var_17 = 'print("Hello")'
    var_18 = 'pre_gen_project'
    var_19 = 'hooks'
    var_20 = 'unsupported_hook.py'
    var_21 = 'print("Hello")'
    var_22 = 'unsupported_hook'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir function.'
    var_1 = 'tests/test-repo'
    var_2 = 'pre_gen_project'
    var_3 = 'tests/test-project'
    var_4 = 'project_name'
    var_5 = 'test-project'
    var_6 = {var_4: var_5}
    var_7 = True



# Parsed testcases at query #31
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = '.'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_script_with_context(var_0, var_1, var_4)
    var_6 = 'invalid_script.py'
    var_7 = module_0.run_script_with_context(var_6, var_1, var_4)
    var_8 = 'test_script.py'
    var_9 = {}
    var_10 = module_0.run_script_with_context(var_8, var_1, var_9)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test_hook'
    var_2 = 'test_project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'test_repo_fail'
    var_8 = 'test_hook_fail'
    var_9 = 'test_project_fail'
    var_10 = {var_3: var_4}
    var_11 = True
    var_12 = 'test_repo_fail_no_delete'
    var_13 = 'test_hook_fail_no_delete'
    var_14 = 'test_project_fail_no_delete'
    var_15 = {var_3: var_4}
    var_16 = False



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'fail_script.py'
    var_3 = 'import sys; sys.exit(1)'
    var_4 = 'non_exec_script.py'
    var_5 = 'print("Hello, World!")'
    var_6 = 420



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test the run_script function.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/sh\necho "Hello, World!"\n'
    var_3 = 493



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'project_name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = "print('Running pre_gen_project hook')"
    var_11 = 'import sys\nsys.exit(1)'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'pre_gen_project'
    var_4 = True
    var_5 = 'hooks'
    var_6 = '.py'
    var_7 = var_3 + var_6
    var_8 = 'print("Hello from hook")'
    var_9 = 'import sys; sys.exit(1)'



# Parsed testcases at query #37
#--------------------------




# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook_from_repo_dir function.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = 'import sys\nsys.exit(1)'
    var_9 = 'pre_gen_project'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello, {{ name }}!')"
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}



# Parsed testcases at query #40
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'print("Hello World")'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py~'
    var_10 = 'print("Hello World")'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py'
    var_14 = 'post_gen_project.py'
    var_15 = 'print("Hello World")'
    var_16 = 'print("Hello World")'
    var_17 = 'pre_gen_project'
    var_18 = 'post_gen_project'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Pre-prompt hook executed")'
    var_3 = 'import sys\nsys.exit(1)'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook_from_repo_dir function.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = 'hooks'
    var_5 = var_2 / var_4
    var_6 = 'pre_gen_project.py'
    var_7 = var_5 / var_6
    var_8 = "print('Hello, world!')"
    var_9 = {}
    var_10 = 'pre_gen_project'
    var_11 = True
    var_12 = 'Project directory should exist after successful hook'
    var_13 = 'import sys; sys.exit(1)'
    var_14 = 'pre_gen_project'
    var_15 = True



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test the run_script function.'
    var_1 = 'test_script.py'
    var_2 = 'print("Hello, world!")'



# Parsed testcases at query #44
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'missing_hooks_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = ''
    var_5 = ''
    var_6 = ''
    var_7 = ''
    var_8 = 'pre_prompt'
    var_9 = 'invalid_hook'
    var_10 = 'pre_prompt.py~'
    var_11 = 'pre_gen_project'



# Parsed testcases at query #45
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the find_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = 'nonexistent_dir'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'hooks'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = ''
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = ''
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'unsupported_hook.py'
    var_16 = ''
    var_17 = 'unsupported_hook'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test the run_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = 'test_project'
    var_3 = 'project_name'
    var_4 = {var_3: var_2}
    var_5 = b"print('Hello, world!')"
    var_6 = b'import sys; sys.exit(1)'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'Test the run_script function.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/sh\necho "Hello, World!"\n'
    var_3 = 493
    var_4 = 'non_existent.sh'
    var_5 = 'empty_script.sh'
    var_6 = ''



# Parsed testcases at query #49
#--------------------------




# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context function.'
    var_1 = 'print("Hello, World!")'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = '.'



