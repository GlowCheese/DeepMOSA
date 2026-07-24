####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'non_existent_hook'
    var_6 = 'pre_gen_project'
    var_7 = 'non_existent_dir'
    var_8 = module_0.find_hook(var_6, var_7)
    assert var_8 is None
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.txt'
    var_11 = 'invalid_hook.py'
    var_12 = 'pre_gen_project.py~'
    var_13 = [var_10, var_11, var_12]
    var_14 = '#!/usr/bin/env python\nprint("test")'
    var_15 = 'pre_gen_project'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = 'hooks'
    var_9 = 'post_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = '{{ undefined_variable }}'
    var_27 = 'project_name'
    var_28 = 'test'
    var_29 = {var_27: var_28}
    var_30 = 'post_gen_project'
    var_31 = False
    var_32 = 'project_name'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = 'post_gen_project'
    var_36 = False



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = var_14 / var_16
    var_18 = 'post_gen_project.py'
    var_19 = var_17 / var_18
    var_20 = 'import sys; sys.exit(1)'
    var_21 = 'project_name'
    var_22 = 'test'
    var_23 = {var_21: var_22}
    var_24 = 'post_gen_project'
    var_25 = False
    var_26 = 'hooks'
    var_27 = var_24 / var_26
    var_28 = 'post_gen_project.py'
    var_29 = var_27 / var_28
    var_30 = '{{ undefined_variable }}'
    var_31 = 'project_name'
    var_32 = 'test'
    var_33 = {var_31: var_32}
    var_34 = 'post_gen_project'
    var_35 = True
    var_36 = 'project_name'
    var_37 = 'test'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True



# Parsed testcases at query #5
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/tmp/repo'
    var_4 = '/fake/repo/hooks/pre_prompt.sh'
    var_5 = '/tmp/repo'
    var_6 = '/fake/repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'project'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.py'
    var_3 = 'print("Hook executed")'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'post_gen_project'
    var_8 = 'nonexistent_hook'
    var_9 = 'invalid_hook.txt'
    var_10 = 'print("This should not execute")'
    var_11 = 'invalid_hook'
    var_12 = 'post_gen_project.py~'
    var_13 = 'post_gen_project'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project.py'
    var_1 = 'pre_gen_project'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True
    var_3 = 'post_gen_project.sh'
    var_4 = 'post_gen_project'
    var_5 = module_0.valid_hook(var_3, var_4)
    assert var_5 is True
    var_6 = module_0.valid_hook(var_0, var_4)
    assert var_6 is False
    var_7 = 'invalid_hook.py'
    var_8 = 'invalid_hook'
    var_9 = module_0.valid_hook(var_7, var_8)
    assert var_9 is False
    var_10 = 'pre_gen_project.py~'
    var_11 = module_0.valid_hook(var_10, var_1)
    assert var_11 is False
    var_12 = 'pre_gen_project.txt'
    var_13 = module_0.valid_hook(var_12, var_1)
    assert var_13 is False



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py'
    var_7 = 'print("Hook executed")'
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'echo "Hello, {{ name }}!"'
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}
    var_5 = 'test_script.py'
    var_6 = 'print("Hello, {{ name }}!")'
    var_7 = {var_2: var_3}
    var_8 = 'test_script_fail.sh'
    var_9 = 'exit 1'
    var_10 = {}
    var_11 = 'test_script_undefined.sh'
    var_12 = 'echo "Hello, {{ undefined_var }}!"'
    var_13 = {}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'cookiecutter.hooks.run_hook'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'cookiecutter.hooks.logger'
    var_10 = 'pre_gen_project'
    var_11 = 'Test error'
    var_12 = 'pre_gen_project'
    var_13 = 'pre_gen_project'
    var_14 = 'pre_gen_project'
    var_15 = False



# Parsed testcases at query #11
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("Valid hook")'
    var_3 = 'pre_gen_project'
    var_4 = 'pre_gen_project'
    var_5 = 'nonexistent'
    var_6 = module_0.find_hook(var_4, var_1)
    assert var_6 is None
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = '#!/usr/bin/env python\nprint("Invalid hook")'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py~'
    var_13 = '#!/usr/bin/env python\nprint("Backup hook")'
    var_14 = 'pre_gen_project'
    var_15 = 'hooks'
    var_16 = 'pre_gen_project.py'
    var_17 = '#!/usr/bin/env python\nprint("Hook 1")'
    var_18 = 'pre_gen_project.sh'
    var_19 = '#!/bin/sh\necho "Hook 2"'
    var_20 = 'pre_gen_project'
    var_21 = len(var_6)
    assert var_21 == 2



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'pre_gen_project.sh'
    var_6 = '#!/bin/sh\necho "test"'
    var_7 = 'invalid_hook.sh'
    var_8 = '#!/bin/sh\necho "test"'
    var_9 = 'pre_gen_project.sh~'
    var_10 = '#!/bin/sh\necho "test"'
    var_11 = 'pre_gen_project.py'
    var_12 = '#!/usr/bin/env python\nprint("test")'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'invalid_hook.py'
    var_3 = 'pre_gen_project.py~'
    var_4 = 'pre_gen_project'
    var_5 = 'non_existent_hook'
    var_6 = 'non_existent_dir'
    var_7 = module_0.find_hook(var_4, var_6)
    assert var_7 is None



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/project_dir'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'
    var_7 = '/fake/hook_script.py'
    var_8 = 'pre_gen_project'
    var_9 = '/fake/project_dir'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.run_hook(var_8, var_9, var_12)
    var_14 = '/fake/hook_script.py'
    var_15 = {var_10: var_11}
    var_16 = 'Running hook %s'
    var_17 = '/fake/hook_script1.py'
    var_18 = '/fake/hook_script2.py'
    var_19 = 'post_gen_project'
    var_20 = '/fake/project_dir'
    var_21 = 'key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = module_0.run_hook(var_19, var_20, var_23)
    var_25 = '/fake/hook_script1.py'
    var_26 = {var_21: var_22}
    var_27 = '/fake/hook_script2.py'
    var_28 = {var_21: var_22}
    var_29 = 'Running hook %s'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'hook_script.py'
    var_6 = 'print("Hook executed")'
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("Valid hook")'
    var_3 = 'pre_gen_project'
    var_4 = 'pre_gen_project'
    var_5 = 'nonexistent'
    var_6 = module_0.find_hook(var_4, var_1)
    assert var_6 is None
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = '#!/usr/bin/env python\nprint("Invalid hook")'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py~'
    var_13 = '#!/usr/bin/env python\nprint("Backup hook")'
    var_14 = 'pre_gen_project'
    var_15 = 'hooks'
    var_16 = 'pre_gen_project.py'
    var_17 = 'pre_gen_project.sh'
    var_18 = '#!/usr/bin/env python\nprint("Hook 1")'
    var_19 = '#!/bin/sh\necho "Hook 2"'
    var_20 = 'pre_gen_project'
    var_21 = len(var_6)
    assert var_21 == 2



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py~'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'invalid_hook.py'
    var_10 = 'pre_gen_project'
    var_11 = 'nonexistent'
    var_12 = 'pre_gen_project'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'fake_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'fake_repo_dir'
    var_2 = 'fake_repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_dir'
    var_4 = 'fake_script.py'
    var_5 = 'temp_dir'
    var_6 = 'fake_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)
    assert var_8 == 'Pre-Prompt Hook script failed'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_prompt'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/fake/tmp/repo'
    var_4 = '/fake/repo'
    var_5 = 'pre_prompt'
    var_6 = '/fake/repo/hooks/pre_prompt.py'
    var_7 = '/fake/tmp/repo'
    var_8 = 'Script failed'
    var_9 = '/fake/repo'
    var_10 = str(var_5)
    assert var_10 == 'Pre-Prompt Hook script failed'
    var_11 = 'pre_prompt'



# Parsed testcases at query #24
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = '#!/usr/bin/env python\nprint("test")'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'pre_gen_project.sh'
    var_11 = '#!/usr/bin/env python\nprint("test")'
    var_12 = '#!/bin/sh\necho "test"'
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py~'
    var_16 = '#!/usr/bin/env python\nprint("test")'
    var_17 = 'pre_gen_project'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = '.sh'
    var_7 = 'echo "Hello, {{ name }}!"'
    var_8 = {var_3: var_4}
    var_9 = 'import sys; sys.exit(1)'
    var_10 = {}



# Parsed testcases at query #26
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_script.py'
    var_3 = 'test_repo'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == 'temp_repo'
    var_5 = 'temp_repo'
    var_6 = 'test_script.py'
    var_7 = 'test error'
    var_8 = 'test_repo'
    var_9 = module_0.run_pre_prompt_hook(var_8)
    var_10 = str(var_5)



# Parsed testcases at query #27
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
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'pre_gen_project.sh'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py~'
    var_14 = 'pre_gen_project'
    var_15 = 'hooks'
    var_16 = 'invalid_hook.py'
    var_17 = 'pre_gen_project'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_3 = None
    var_4 = '/fake/repo_dir'
    var_5 = module_0.run_pre_prompt_hook(var_4)
    assert var_5 == '/tmp/fake_repo'
    var_6 = '/tmp/fake_repo'
    var_7 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_8 = None
    var_9 = 'Hook failed'
    var_10 = '/fake/repo_dir'
    var_11 = module_0.run_pre_prompt_hook(var_10)
    var_12 = str(var_6)
    assert var_12 == 'Pre-Prompt Hook script failed'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed successfully")'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'project_name'
    var_20 = 'test_project'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = '{{ undefined_variable }}'
    var_27 = 'project_name'
    var_28 = 'test_project'
    var_29 = {var_27: var_28}
    var_30 = 'post_gen_project'
    var_31 = True
    var_32 = 'project_name'
    var_33 = 'test_project'
    var_34 = {var_32: var_33}
    var_35 = 'non_existent_hook'
    var_36 = True



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = 'invalid_hook.py'
    var_5 = 'pre_gen_project'
    var_6 = 'non_existent_hooks_dir'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'pre_gen_project.sh'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py~'
    var_14 = 'pre_gen_project'



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 'hooks'
    var_7 = 'post_gen_project.sh'
    var_8 = "#!/bin/sh\necho 'test'"
    var_9 = 'cookiecutter.hooks.work_in'
    var_10 = lambda x: x
    var_11 = 'cookiecutter.hooks.rmtree'
    var_12 = 'post_gen_project'
    var_13 = '#!/bin/sh\nexit 1'
    var_14 = 'post_gen_project'
    var_15 = "#!/bin/sh\necho '{{ undefined_variable }}'"
    var_16 = 'post_gen_project'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'echo "Hello, World!"'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py~'
    var_7 = 'pre_gen_project'
    var_8 = 'hooks'
    var_9 = 'wrong_name.py'
    var_10 = 'pre_gen_project'
    var_11 = 'nonexistent'
    var_12 = 'pre_gen_project'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/repo'
    var_2 = '/fake/tmp/repo'
    var_3 = '/fake/repo/hooks/pre_prompt.sh'
    var_4 = '/fake/repo'



# Parsed testcases at query #36
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'hooks'
    var_4 = '# invalid hook'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = '# valid hook'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py'
    var_12 = 'pre_gen_project.sh'
    var_13 = '# valid hook 1'
    var_14 = '# valid hook 2'
    var_15 = 'pre_gen_project'
    assert var_15 is None
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py~'
    var_18 = '# backup hook'
    var_19 = 'pre_gen_project'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = '/path/to/script.py'
    var_7 = [var_6]
    var_8 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("Hello")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'import sys; sys.exit(1)'



# Parsed testcases at query #39
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'failing_script.sh'
    var_3 = '#!/bin/sh\nexit 1\n'
    var_4 = 'test_python.py'
    var_5 = 'print("Python script executed")\n'
    var_6 = 'no_shebang.sh'
    var_7 = 'echo "No shebang"\n'
    var_8 = '/nonexistent/script.sh'
    var_9 = module_0.run_script(var_8)
    var_10 = str(var_8)



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'test_script.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = 'subprocess.Popen'
    var_7 = 'cookiecutter.utils.make_executable'
    var_8 = 'cookiecutter.hooks.create_env_with_context'
    var_9 = False



# Parsed testcases at query #41
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '#!/bin/sh\necho "Hello, World!"'
    var_3 = '/path/to/nonexistent/script.py'
    var_4 = module_0.run_script(var_3)
    var_5 = ''



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = '/path/to/script.py'
    var_7 = [var_6]
    var_8 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello"\n'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\nexit 1\n'
    var_4 = 'test_script.py'
    var_5 = 'print("Hello")\n'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello"\n'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = 'echo "Hello, World!"\n'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = '/path/to/script.py'
    var_7 = [var_6]
    var_8 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None
    var_7 = 'cookiecutter.hooks.run_script_with_context'

def test_case_0():
    var_0 = 'post_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = '/path/to/script1.py'
    var_6 = '/path/to/script2.sh'
    var_7 = [var_5, var_6]
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = 'cookiecutter.hooks.run_script_with_context'
    var_10 = 0
    var_11 = var_7[var_10]
    var_12 = 1
    var_13 = var_7[var_12]
    var_14 = False



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo_dir'
    var_2 = 'test_repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo_dir'
    var_4 = 'test_script.py'
    var_5 = 'temp_repo_dir'
    var_6 = 'test_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = module_0.find_hook(var_2)
    var_4 = 'pre_gen_project'
    var_5 = module_0.find_hook(var_4)
    assert var_5 is None
    var_6 = 'hooks'
    var_7 = 'invalid_hook.py'
    var_8 = 'pre_gen_project'
    var_9 = module_0.find_hook(var_8)
    assert var_9 is None
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = module_0.find_hook(var_12)
    assert var_13 is None



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'utf-8'
    var_3 = '.'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'project_name'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = 'post_gen_project'
    var_28 = True



# Parsed testcases at query #5
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/fake/repo'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == '/tmp/repo'
    var_5 = '/tmp/repo'
    var_6 = '/fake/repo/hooks/pre_prompt.py'
    var_7 = 'Script failed'
    var_8 = '/fake/repo'
    var_9 = module_0.run_pre_prompt_hook(var_8)
    var_10 = str(var_5)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = 'This is not executable'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'failing_script.sh'
    var_3 = '#!/bin/sh\nexit 1\n'
    var_4 = 'test_script.py'
    var_5 = 'print("Hello, Python!")\n'
    var_6 = 'noshebang_script.sh'
    var_7 = 'echo "No shebang"\n'
    var_8 = 'nonexistent_script.sh'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'test_script.py'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'test_script.py'
    var_3 = 'print("Hello, World!")\n'
    var_4 = 'test_script.sh'
    var_5 = '#!/bin/sh\nexit 1\n'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"\n'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'No %s hook found'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Hook executed")'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = {var_4: var_5}
    var_13 = 'exit(1)'
    var_14 = 'Hook failed'
    var_15 = 'pre_gen_project'
    var_16 = 'project_name'
    var_17 = 'test'
    var_18 = {var_16: var_17}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'No %s hook found'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Hook executed")'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = {var_4: var_5}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger.debug'
    var_3 = 'pre_gen_project'
    var_4 = {}
    var_5 = 'No %s hook found'
    var_6 = 'hook.py'
    var_7 = 'print("test")'
    var_8 = 'cookiecutter.hooks.run_script_with_context'
    var_9 = 'test'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'pre_gen_project'
    var_4 = 'nonexistent'
    var_5 = module_0.find_hook(var_3, var_1)
    assert var_5 is None
    var_6 = 'hooks'
    var_7 = 'other_script.py'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.py~'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'unsupported_hook.py'
    var_14 = 'pre_gen_project'



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = 'Expected FailedHookException for non-zero exit status'
    var_3 = 'echo "Hello, World!"'
    var_4 = 'Expected FailedHookException for missing shebang'
    var_5 = '#!/bin/bash\necho "Hello, World!"'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_3 = '/fake/repo_dir'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == '/tmp/fake_repo'
    var_5 = '/tmp/fake_repo'
    var_6 = '/fake/repo_dir/hooks/pre_prompt.py'
    var_7 = 'Test error'
    var_8 = '/fake/repo_dir'
    var_9 = module_0.run_pre_prompt_hook(var_8)
    var_10 = str(var_5)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '#!/bin/sh\necho "Hello, World!"'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/project/dir'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'
    var_7 = '/fake/hook/script.py'
    var_8 = 'pre_gen_project'
    var_9 = '/fake/project/dir'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.run_hook(var_8, var_9, var_12)
    var_14 = '/fake/hook/script.py'
    var_15 = {var_10: var_11}
    var_16 = '/fake/hook/script1.py'
    var_17 = '/fake/hook/script2.sh'
    var_18 = 'post_gen_project'
    var_19 = '/fake/project/dir'
    var_20 = 'key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = module_0.run_hook(var_18, var_19, var_22)
    var_24 = '/fake/hook/script1.py'
    var_25 = {var_20: var_21}
    var_26 = '/fake/hook/script2.sh'
    var_27 = {var_20: var_21}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = '.'
    var_7 = "echo 'Hello, {{ name }}!'"
    var_8 = '.sh'
    var_9 = "print('Hello, {{ missing_var }}!')"
    var_10 = {var_3: var_4}
    var_11 = '.'
    var_12 = 'import sys; sys.exit(1)'
    var_13 = {var_3: var_4}
    var_14 = '.'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.py'
    var_5 = 'utf-8'
    var_6 = '.'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/tmp/repo_dir'
    var_4 = '/tmp/repo_dir'
    var_5 = '/fake/script.py'
    var_6 = '/fake/repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 493
    var_3 = 'test_script.py'
    var_4 = 'print("Hello, World!")\n'
    var_5 = 'test_script.sh'
    var_6 = '#!/bin/sh\nexit 1\n'
    var_7 = 493
    var_8 = 'test_script.sh'
    var_9 = 'echo "Hello, World!"\n'
    var_10 = str(var_7)
    var_11 = 'test_script.sh'
    var_12 = '#!/bin/sh\necho "Hello, World!"\n'
    var_13 = 493
    var_14 = 'Permission denied'
    var_15 = str(var_10)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_prompt'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/fake/tmp/repo'
    var_4 = '/fake/repo'
    var_5 = 'pre_prompt'
    var_6 = '/fake/repo/hooks/pre_prompt.py'
    var_7 = '/fake/tmp/repo'
    var_8 = 'Test failure'
    var_9 = '/fake/repo'
    var_10 = str(var_5)
    assert var_10 == 'Pre-Prompt Hook script failed'
    var_11 = 'pre_prompt'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger.debug'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'No %s hook found'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Hook executed")'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = {var_4: var_5}
    var_13 = {var_4: var_5}



# Parsed testcases at query #27
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/project_dir'
    var_2 = 'fake'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/fake/hook_script.sh'
    var_7 = 'post_gen_project'
    var_8 = '/fake/project_dir'
    var_9 = 'fake'
    var_10 = 'context'
    var_11 = {var_9: var_10}
    var_12 = module_0.run_hook(var_7, var_8, var_11)
    var_13 = {var_9: var_10}
    var_14 = '/fake/hook1.sh'
    var_15 = '/fake/hook2.py'
    var_16 = 'pre_gen_project'
    var_17 = '/fake/project_dir'
    var_18 = 'fake'
    var_19 = 'context'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_hook(var_16, var_17, var_20)
    var_22 = {var_18: var_19}
    var_23 = {var_18: var_19}



# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'test'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'test'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'test'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'test'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = 'post_gen_project'
    var_28 = True



