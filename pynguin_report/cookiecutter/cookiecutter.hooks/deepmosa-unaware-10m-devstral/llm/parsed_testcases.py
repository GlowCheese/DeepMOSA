####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'hooks'
    var_4 = '#!/usr/bin/env python\nprint("Hello")'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = '#!/usr/bin/env python\nprint("Hello")'
    var_8 = 'pre_gen_project'
    var_9 = 0
    var_10 = 'pre_gen_project.py'
    var_11 = 'hooks'
    var_12 = '#!/usr/bin/env python\nprint("Hello")'
    var_13 = '#!/bin/sh\necho "Hello"'
    var_14 = 'pre_gen_project'
    var_15 = 'pre_gen_project.py'
    var_16 = 'pre_gen_project.sh'
    var_17 = (var_15, var_16)
    var_18 = 'hooks'
    var_19 = '#!/usr/bin/env python\nprint("Hello")'
    var_20 = '#!/usr/bin/env python\nprint("Hello")'
    var_21 = 'pre_gen_project'
    var_22 = 0
    var_23 = 'pre_gen_project.py'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'
    var_8 = 'test_script.py'
    var_9 = 'import sys\nsys.exit(0)'
    var_10 = 0



# Parsed testcases at query #4
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/tmp/repo'
    var_4 = '/fake/repo/hooks/pre_prompt.py'
    var_5 = '/tmp/repo'
    var_6 = '/fake/repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)



# Parsed testcases at query #5
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
    var_13 = {var_4: var_5}
    var_14 = 'pre_gen_project.sh'
    var_15 = 'print("Hook 1 executed")'
    var_16 = 'echo "Hook 2 executed"'
    var_17 = {var_4: var_5}



# Parsed testcases at query #6
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '/non/existent/script.py'
    var_3 = module_0.run_script(var_2)
    var_4 = ''



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
    var_6 = 'invalid_hook.py'
    var_7 = module_0.valid_hook(var_6, var_1)
    assert var_7 is False
    var_8 = 'invalid_hook'
    var_9 = module_0.valid_hook(var_6, var_8)
    assert var_9 is False
    var_10 = 'pre_gen_project.py~'
    var_11 = module_0.valid_hook(var_10, var_1)
    assert var_11 is False
    var_12 = 'pre_prompt.sh'
    var_13 = 'pre_prompt'
    var_14 = module_0.valid_hook(var_12, var_13)
    assert var_14 is True
    var_15 = module_0.valid_hook(var_13, var_13)
    assert var_15 is True



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = b'print("Hello, World!")'
    var_1 = b'import sys; sys.exit(1)'
    var_2 = str(var_1)
    var_3 = '/non/existent/script.py'
    var_4 = module_0.run_script(var_3)
    var_5 = str(var_3)
    var_6 = b'print("Hello, World!")'
    var_7 = str(var_6)



# Parsed testcases at query #9
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
    var_9 = 'invalid_hook.py'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py~'
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py'
    var_16 = 'pre_gen_project.sh'
    var_17 = 'pre_gen_project'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'
    var_8 = 'test_script.py'
    var_9 = 'import sys\nsys.exit(0)'
    var_10 = 'Test error'
    var_11 = 'Test error'



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = None
    var_4 = '/fake/repo'
    var_5 = module_0.run_pre_prompt_hook(var_4)
    assert var_5 == '/tmp/fake_repo'
    var_6 = '/tmp/fake_repo'
    var_7 = '/fake/repo/hooks/pre_prompt.py'
    var_8 = None
    var_9 = 'Test error'
    var_10 = '/fake/repo'
    var_11 = module_0.run_pre_prompt_hook(var_10)
    var_12 = str(var_6)
    assert var_12 == 'Pre-Prompt Hook script failed'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho "Hello, World!"\n'
    var_2 = 'subprocess.Popen'
    var_3 = module_0.run_script(var_0)
    var_4 = 'test_script.py'
    var_5 = 'print("Hello, World!")\n'
    var_6 = 'Test error'
    var_7 = module_0.run_script(var_0)
    var_8 = module_0.run_script(var_0)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = {}
    var_2 = 'hooks'
    var_3 = 'pre_gen_project.py'
    var_4 = 'print("Hook executed")'
    var_5 = 'pre_gen_project'
    var_6 = {}
    var_7 = 'hooks'
    var_8 = 'post_gen_project.sh'
    var_9 = 'echo "Project name is {{ cookiecutter.project_name }}"'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = 'post_gen_project'
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'pre_gen_project'
    var_20 = {}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = 'pre_prompt'
    var_3 = '/fake/script.py'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = '/fake/repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    assert var_7 == '/tmp/repo'
    var_8 = '/tmp/repo'
    assert var_8 == 'Pre-Prompt Hook script failed'
    var_9 = '/fake/script.py'
    var_10 = [var_9]
    var_11 = [var_9]
    var_12 = 'Failed'
    var_13 = '/fake/repo'
    var_14 = module_0.run_pre_prompt_hook(var_13)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = 'echo "Hello, {{ name }}!"'
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}
    var_5 = 'test_py_script.py'
    var_6 = 'print("Hello, {{ name }}!")'
    var_7 = 'failing_script.sh'
    var_8 = 'exit 1'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = None
    var_4 = 'hooks'
    var_5 = 'pre_prompt.py'
    var_6 = "#!/usr/bin/env python\nprint('Hello')"
    var_7 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_10 = 'Failed'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '#!/bin/sh\necho "Hello, World!"\n'
    var_1 = 'print("Hello, World!")\n'
    var_2 = '#!/bin/sh\nexit 1\n'
    var_3 = str(var_2)
    var_4 = '#!/bin/sh\nnonexistent_command\n'
    var_5 = str(var_4)
    var_6 = 'echo "Hello, World!"\n'
    var_7 = str(var_6)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.py'
    var_4 = 'print("Hook executed")'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.py'
    var_14 = 'import sys; sys.exit(1)'
    var_15 = 'project_name'
    var_16 = 'test'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.py'
    var_26 = var_24 / var_25
    var_27 = 'import sys; sys.exit(1)'
    var_28 = 'project_name'
    var_29 = 'test'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'project_name'
    var_37 = 'test'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True



# Parsed testcases at query #22
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo_dir'
    var_2 = 'pre_prompt'
    var_3 = 'pre_prompt_script'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 'test_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    assert var_7 == 'temp_repo_dir'
    var_8 = 'pre_prompt'
    var_9 = 'temp_repo_dir'
    var_10 = 'pre_prompt_script'
    var_11 = [var_10]
    var_12 = [var_10]
    var_13 = 'Test error'
    var_14 = 'test_repo_dir'
    var_15 = module_0.run_pre_prompt_hook(var_14)
    var_16 = str(var_8)
    assert var_16 == 'Pre-Prompt Hook script failed'
    var_17 = 'pre_prompt'
    var_18 = 'test_repo_dir'
    var_19 = 'temp_repo_dir'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/sh\necho "Hook executed"\n'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.sh'
    var_10 = '#!/bin/sh\nexit 1\n'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.sh'
    var_18 = '#!/bin/sh\nexit 1\n'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'project_name'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = 'non_existent_hook'
    var_28 = True



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger.debug'
    var_3 = 'pre_gen_project'
    var_4 = {}
    var_5 = 'No %s hook found'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("Hello")'
    var_9 = 'cookiecutter.hooks.run_script_with_context'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = {var_10: var_11}
    var_14 = 'pre_gen_project.sh'
    var_15 = 'echo "Hello"'
    var_16 = {var_10: var_11}
    var_17 = {var_10: var_11}
    var_18 = {var_10: var_11}



# Parsed testcases at query #27
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
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = '{{ undefined_variable }}'
    var_27 = 'project_name'
    var_28 = 'test'
    var_29 = {var_27: var_28}
    var_30 = 'post_gen_project'
    var_31 = True
    var_32 = 'project_name'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = 'post_gen_project'
    var_36 = True



# Parsed testcases at query #28
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'script1.py'
    var_3 = 'script2.sh'
    var_4 = None
    var_5 = 'test_repo'
    var_6 = module_0.run_pre_prompt_hook(var_5)
    assert var_6 == 'temp_repo'
    var_7 = 'script1.py'
    var_8 = None
    var_9 = 'Test error'
    var_10 = 'test_repo'
    var_11 = module_0.run_pre_prompt_hook(var_10)



# Parsed testcases at query #29
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
    var_9 = ''
    var_10 = 493
    var_11 = 'test_script.sh'
    var_12 = 'echo "Hello, World!"\n'
    var_13 = 493



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_prompt'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = '/fake/tmp/repo'
    var_6 = '/fake/repo'
    var_7 = 'pre_prompt'
    var_8 = '/fake/repo/hooks/pre_prompt.py'
    var_9 = [var_8]
    var_10 = [var_8]
    var_11 = '/fake/tmp/repo'
    var_12 = 'Script failed'
    var_13 = '/fake/repo'
    var_14 = str(var_7)
    assert var_14 == 'Pre-Prompt Hook script failed'
    var_15 = 'pre_prompt'



# Parsed testcases at query #31
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/tmp/hook_script.py'
    var_7 = 'pre_gen_project'
    var_8 = '/tmp/project'
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = module_0.run_hook(var_7, var_8, var_11)
    var_13 = {var_9: var_10}
    var_14 = '/tmp/hook_script1.py'
    var_15 = '/tmp/hook_script2.py'
    var_16 = 'post_gen_project'
    var_17 = '/tmp/project'
    var_18 = 'name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_hook(var_16, var_17, var_20)
    var_22 = {var_18: var_19}
    var_23 = {var_18: var_19}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project'
    var_2 = '#!/bin/sh\necho "Hook executed"'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'hooks'
    var_8 = 'pre_gen_project'
    var_9 = '#!/bin/sh\nexit 1'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = 'pre_gen_project'
    var_14 = True
    var_15 = 'hooks'
    var_16 = 'pre_gen_project'
    var_17 = '#!/bin/sh\nexit 1'
    var_18 = 'project_name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = 'pre_gen_project'
    var_22 = False
    var_23 = 'project_name'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = 'pre_gen_project'
    var_27 = True



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project'
    var_2 = '#!/bin/sh\necho "Hook executed"'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'hooks'
    var_8 = 'post_gen_project'
    var_9 = '#!/bin/sh\nexit 1'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'hooks'
    var_16 = 'post_gen_project'
    var_17 = '#!/bin/sh\nexit 1'
    var_18 = 'project_name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = 'post_gen_project'
    var_22 = False
    var_23 = 'project_name'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = 'post_gen_project'
    var_27 = True



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'pre_prompt'
    var_2 = '/tmp/test_repo/hooks/pre_prompt.py'
    var_3 = '/tmp/test_repo_temp'
    var_4 = '/tmp/test_repo'
    var_5 = 'pre_prompt'
    var_6 = '/tmp/test_repo_temp/hooks/pre_prompt.py'
    var_7 = '/tmp/test_repo/hooks/pre_prompt.py'
    var_8 = '/tmp/test_repo_temp'
    var_9 = 'Test error'
    var_10 = '/tmp/test_repo'
    var_11 = str(var_5)



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = '#!/bin/sh\necho "test"\n'
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.sh'
    var_14 = '#!/bin/sh\nexit 1\n'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.sh'
    var_26 = var_24 / var_25
    var_27 = '#!/bin/sh\nexit 1\n'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #38
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'pre_gen_project'



# Parsed testcases at query #39
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



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = '#!/bin/sh\nexit 0'
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.sh'
    var_14 = '#!/bin/sh\nexit 1'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.sh'
    var_26 = var_24 / var_25
    var_27 = '#!/bin/sh\nexit 1'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True



# Parsed testcases at query #41
#--------------------------




# Parsed testcases at query #42
#--------------------------




# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project'
    var_2 = '#!/bin/bash\necho "Hook executed"'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'hooks'
    var_8 = 'post_gen_project'
    var_9 = '#!/bin/bash\nexit 1'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'hooks'
    var_16 = 'post_gen_project'
    var_17 = '#!/bin/bash\nexit 1'
    var_18 = 'project_name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = 'post_gen_project'
    var_22 = False
    var_23 = 'hooks'
    var_24 = 'post_gen_project.py'
    var_25 = '{{ undefined_variable }}'
    var_26 = 'project_name'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = 'post_gen_project'
    var_30 = True
    var_31 = 'project_name'
    var_32 = 'test'
    var_33 = {var_31: var_32}
    var_34 = 'post_gen_project'
    var_35 = True



# Parsed testcases at query #44
#--------------------------




# Parsed testcases at query #45
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
    var_7 = 'Hook failed'
    var_8 = '/fake/repo'
    var_9 = module_0.run_pre_prompt_hook(var_8)
    var_10 = str(var_5)



# Parsed testcases at query #46
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = 'name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = '/tmp/hook.py'
    var_13 = {var_8: var_9}
    var_14 = 'post_gen_project'
    var_15 = '/tmp/project'
    var_16 = 'name'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = module_0.run_hook(var_14, var_15, var_18)
    var_20 = '/tmp/hook1.py'
    var_21 = {var_16: var_17}
    var_22 = '/tmp/hook2.py'
    var_23 = {var_16: var_17}



# Parsed testcases at query #47
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
    var_8 = 'test_script.sh'
    var_9 = '#!/bin/sh\nnonexistent_command\n'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = "#!/bin/sh\necho 'Hello'\n"
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.sh'
    var_14 = '#!/bin/sh\nexit 1\n'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.sh'
    var_26 = var_24 / var_25
    var_27 = '#!/bin/sh\nexit 1\n'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = False



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.work_in'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = 'pre_prompt.py'
    var_4 = '#!/usr/bin/env python\nprint("test")'
    var_5 = 'tmp'
    var_6 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 'test error'



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/repo/hooks/pre_prompt.py'
    var_2 = '/fake/tmp/repo'
    var_3 = None
    var_4 = '/fake/repo'
    var_5 = '/fake/repo/hooks/pre_prompt.py'
    var_6 = '/fake/tmp/repo'
    var_7 = None
    var_8 = 'Hook failed'
    var_9 = '/fake/repo'



# Parsed testcases at query #52
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
    var_24 = 'project_name'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = 'post_gen_project'
    var_28 = False



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.py'
    var_3 = "print('Hook executed')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'cookiecutter.utils.work_in'
    var_9 = lambda x: x
    var_10 = 'post_gen_project'
    var_11 = True
    var_12 = 'import sys; sys.exit(1)'
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'post_gen_project'
    var_16 = False
    var_17 = 'non_existent_hook'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger.debug'
    var_3 = 'pre_gen_project'
    var_4 = {}
    var_5 = 'No %s hook found'
    var_6 = 'hook.py'
    var_7 = 'print("Hello")'
    var_8 = 'cookiecutter.hooks.run_script_with_context'
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}
    var_13 = 'hook2.py'
    var_14 = 'print("World")'
    var_15 = {var_9: var_10}



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = '#!/bin/sh\necho "test"\n'
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.sh'
    var_14 = '#!/bin/sh\nexit 1\n'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.sh'
    var_26 = var_24 / var_25
    var_27 = '#!/bin/sh\nexit 1\n'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True
    var_41 = 'repo'
    var_42 = var_31 / var_41
    var_43 = 'project'
    var_44 = 'hooks'
    var_45 = var_42 / var_44
    var_46 = 'post_gen_project.py'
    var_47 = var_45 / var_46
    var_48 = '{{ undefined_var }}\n'
    var_49 = 'test'
    var_50 = 'value'
    var_51 = {var_49: var_50}
    var_52 = 'post_gen_project'
    var_53 = True



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = 'script1.py'
    var_2 = 'script2.sh'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'test_project'
    var_6 = 'project_name'
    var_7 = {var_6: var_5}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.run_script_with_context'
    var_3 = 'test_project'
    var_4 = 'project_name'
    var_5 = {var_4: var_3}
    var_6 = 'pre_gen_project'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'
    var_8 = 'test_script.py'
    var_9 = 'import sys\nsys.exit(0)'
    var_10 = 'File not found'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #60
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/tmp/hooks/pre_gen_project.sh'
    var_7 = 'pre_gen_project'
    var_8 = '/tmp/project'
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = module_0.run_hook(var_7, var_8, var_11)
    var_13 = {var_9: var_10}
    var_14 = '/tmp/hooks/pre_gen_project.sh'
    var_15 = '/tmp/hooks/pre_gen_project.py'
    var_16 = 'pre_gen_project'
    var_17 = '/tmp/project'
    var_18 = 'name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_hook(var_16, var_17, var_20)
    var_22 = {var_18: var_19}
    var_23 = {var_18: var_19}



# Parsed testcases at query #61
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
    var_27 = 'non_existent_hook'
    var_28 = True
    var_29 = 'hooks'
    var_30 = 'post_gen_project.py'
    var_31 = '{{ undefined_variable }}'
    var_32 = 'project_name'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = 'post_gen_project'
    var_36 = True



# Parsed testcases at query #62
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'subprocess.Popen'
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = False
    var_5 = '.'
    var_6 = 'failing_script.sh'
    var_7 = '#!/bin/sh\nexit 1\n'
    var_8 = module_0.run_script(var_0)
    var_9 = 'test_python.py'
    var_10 = 'print("Hello from Python")\n'
    var_11 = 'sys.platform'
    var_12 = 'win32'
    var_13 = 'test_script.bat'
    var_14 = 'echo Hello from Windows\n'
    var_15 = True
    var_16 = 'empty_script.sh'
    var_17 = ''
    var_18 = 'No exec'
    var_19 = module_0.run_script(var_0)
    var_20 = 'Permission denied'
    var_21 = module_0.run_script(var_0)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/tmp/repo'
    var_4 = '/fake/hook.py'
    var_5 = '/tmp/repo'
    var_6 = '/fake/repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'test'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
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
    var_27 = 'non_existent_hook'
    var_28 = False



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #6
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
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'pre_gen_project'



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
    var_8 = module_0.valid_hook(var_7, var_1)
    assert var_8 is False
    var_9 = 'pre_gen_project.py~'
    var_10 = module_0.valid_hook(var_9, var_1)
    assert var_10 is False
    var_11 = 'invalid_hook'
    var_12 = module_0.valid_hook(var_7, var_11)
    assert var_12 is False



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'no_hook'
    var_1 = 'with_hook'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = "#!/usr/bin/env python\nprint('Pre-prompt hook executed')"
    var_5 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_6 = 'cookiecutter.hooks.work_in'
    var_7 = 'failing_hook'
    var_8 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.py'
    var_4 = "print('Hook executed')"
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.py'
    var_14 = 'import sys; sys.exit(1)'
    var_15 = 'project_name'
    var_16 = 'test'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.py'
    var_26 = var_24 / var_25
    var_27 = 'import sys; sys.exit(1)'
    var_28 = 'project_name'
    var_29 = 'test'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'project_name'
    var_37 = 'test'
    var_38 = {var_36: var_37}
    var_39 = 'post_gen_project'
    var_40 = True



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/bash\necho "Hello, World!"\n'
    var_2 = 'bad_script.sh'
    var_3 = '#!/bin/bash\nexit 1\n'
    var_4 = 'no_shebang.sh'
    var_5 = 'echo "No shebang"\n'
    var_6 = 'test_script.py'
    var_7 = 'print("Hello from Python")\n'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = "echo 'Hello, {{ name }}!'"
    var_7 = '.sh'
    var_8 = {var_3: var_4}
    var_9 = 'exit 1'
    var_10 = {}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #6
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'hook_script'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



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
    var_6 = 'invalid_hook.py'
    var_7 = module_0.valid_hook(var_6, var_1)
    assert var_7 is False
    var_8 = 'invalid_hook'
    var_9 = module_0.valid_hook(var_0, var_8)
    assert var_9 is False
    var_10 = module_0.valid_hook(var_6, var_8)
    assert var_10 is False
    var_11 = 'pre_gen_project.py~'
    var_12 = module_0.valid_hook(var_11, var_1)
    assert var_12 is False



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
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
    var_7 = 'other_hook.py'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.py~'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py'
    var_14 = 'pre_gen_project.sh'
    var_15 = 'pre_gen_project'
    var_16 = len(var_5)
    assert var_16 == 2



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
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.py'
    var_4 = "print('Hook executed')"
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.py'
    var_14 = 'import sys; sys.exit(1)'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.py'
    var_26 = var_24 / var_25
    var_27 = 'import sys; sys.exit(1)'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'hooks'
    var_37 = var_34 / var_36
    var_38 = 'post_gen_project.py'
    var_39 = var_37 / var_38
    var_40 = '{{ undefined_variable }}'
    var_41 = 'test'
    var_42 = 'value'
    var_43 = {var_41: var_42}
    var_44 = 'post_gen_project'
    var_45 = True
    var_46 = 'repo'
    var_47 = var_44 / var_46
    var_48 = 'project'
    var_49 = 'test'
    var_50 = 'value'
    var_51 = {var_49: var_50}
    var_52 = 'post_gen_project'
    var_53 = False



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'pre_prompt'
    var_4 = 'hooks'
    var_5 = 'pre_prompt.py'
    var_6 = 'pre_prompt'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_prompt'
    var_10 = 'hooks'
    var_11 = 'pre_prompt.py~'
    var_12 = 'pre_prompt'
    var_13 = 'hooks'
    var_14 = 'pre_prompt.py'
    var_15 = 'pre_prompt.sh'
    var_16 = 'pre_prompt'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/script.py'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = None
    var_6 = '/fake/repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    assert var_7 == '/fake/tmp_repo_dir'
    var_8 = '/fake/tmp_repo_dir'
    var_9 = '/fake/script.py'
    var_10 = [var_9]
    var_11 = [var_9]
    var_12 = None
    var_13 = 'Test error'
    var_14 = '/fake/repo_dir'
    var_15 = module_0.run_pre_prompt_hook(var_14)
    var_16 = str(var_8)



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'script.sh'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.py'
    var_3 = "print('success')"
    var_4 = 'project'
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = 'repo'
    var_11 = 'hooks'
    var_12 = 'post_gen_project.py'
    var_13 = 'import sys; sys.exit(1)'
    var_14 = 'project'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'hooks'
    var_23 = var_21 / var_22
    var_24 = 'post_gen_project.py'
    var_25 = var_23 / var_24
    var_26 = 'import sys; sys.exit(1)'
    var_27 = 'project'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = var_22 / var_35
    var_37 = 'test'
    var_38 = 'value'
    var_39 = {var_37: var_38}
    var_40 = 'post_gen_project'
    var_41 = False
    var_42 = module_0.run_hook_from_repo_dir(var_34, var_40, var_36, var_39, var_41)
    var_43 = 'repo'
    var_44 = var_31 / var_43
    var_45 = 'hooks'
    var_46 = var_44 / var_45
    var_47 = 'post_gen_project.py'
    var_48 = var_46 / var_47
    var_49 = '{{ undefined_var }}'
    var_50 = 'project'
    var_51 = var_41 / var_50
    var_52 = 'test'
    var_53 = 'value'
    var_54 = {var_52: var_53}
    var_55 = 'post_gen_project'
    var_56 = True
    var_57 = module_0.run_hook_from_repo_dir(var_44, var_55, var_51, var_54, var_56)



# Parsed testcases at query #17
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
    var_24 = 'project_name'
    var_25 = 'test_project'
    var_26 = {var_24: var_25}
    var_27 = 'post_gen_project'
    var_28 = True



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #19
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'hook_script'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #21
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'script.sh'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #23
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



# Parsed testcases at query #24
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
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'pre_gen_project'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #26
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/fake/tmp_repo'
    var_4 = '/fake/script.py'
    var_5 = '/fake/tmp_repo'
    var_6 = '/fake/repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'post_gen_project'
    var_4 = {}
    var_5 = False
    var_6 = 'hooks'
    var_7 = 'post_gen_project.py'
    var_8 = 'import sys; sys.exit(1)'
    var_9 = 'post_gen_project'
    var_10 = {}
    var_11 = True
    var_12 = 'hooks'
    var_13 = 'post_gen_project.py'
    var_14 = 'import sys; sys.exit(1)'
    var_15 = 'post_gen_project'
    var_16 = {}
    var_17 = False
    var_18 = 'post_gen_project'
    var_19 = {}
    var_20 = False



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'invalid_hook.py'
    var_7 = 'pre_gen_project.py~'
    var_8 = 'pre_gen_project'
    var_9 = 'nonexistent_hooks_dir'
    var_10 = 'pre_gen_project'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #32
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
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'pre_gen_project.sh'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py~'
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'invalid_hook.py'
    var_16 = 'pre_gen_project'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.py'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'pre_gen_project'
    var_7 = '/fake/project'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = '/fake/hook.py'
    var_13 = {var_8: var_9}
    var_14 = 'pre_gen_project'
    var_15 = '/fake/project'
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = module_0.run_hook(var_14, var_15, var_18)
    var_20 = '/fake/hook1.py'
    var_21 = {var_16: var_17}
    var_22 = '/fake/hook2.py'
    var_23 = {var_16: var_17}



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'pre_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = 'pre_gen_project'
    var_23 = False
    var_24 = 'project_name'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = 'pre_gen_project'
    var_28 = True



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = '#!/bin/sh\necho "Hello"\n'
    var_1 = '#!/bin/sh\nexit 1\n'
    var_2 = 'print("Hello")\n'
    var_3 = 'echo "Hello"\n'



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
    var_27 = 'non_existent_hook'
    var_28 = True



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'hooks'
    var_4 = 'post_gen_project.py'
    var_5 = 'print("Hook executed")'
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'import sys; sys.exit(1)'
    var_9 = 'post_gen_project'
    var_10 = True
    var_11 = 'Expected FailedHookException'
    var_12 = 'post_gen_project'
    var_13 = False
    var_14 = 'Expected FailedHookException'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, {{ name }}!"'
    var_8 = 'utf-8'
    var_9 = 'name'
    var_10 = 'World'
    var_11 = {var_9: var_10}
    var_12 = 'test_script.py'
    var_13 = 'import sys; sys.exit(1)'
    var_14 = 'utf-8'
    var_15 = {}
    var_16 = 'test_script.py'
    var_17 = 'print("Hello, {{ undefined_var }}!")'
    var_18 = 'utf-8'
    var_19 = {}



# Parsed testcases at query #41
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '#!/bin/sh\necho "Hello, World!"\n'
    var_1 = '#!/bin/sh\nexit 1\n'
    var_2 = str(var_1)
    var_3 = 'print("Hello, World!")\n'
    var_4 = '/non/existent/script.sh'
    var_5 = module_0.run_script(var_4)
    var_6 = str(var_4)
    var_7 = ''
    var_8 = str(var_7)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("Valid hook")'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py~'
    var_8 = '#!/usr/bin/env python\nprint("Backup hook")'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'wrong_hook.py'
    var_12 = '#!/usr/bin/env python\nprint("Wrong hook")'
    var_13 = 'pre_gen_project'
    var_14 = 'nonexistent_hooks_dir'
    var_15 = 'pre_gen_project'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = '/fake/repo'
    var_2 = '/fake/tmp/repo'
    var_3 = '/fake/repo/hooks/pre_prompt.py'
    var_4 = '/fake/repo'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'print("Hook executed")'
    var_3 = 'test'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'import sys; sys.exit(1)'
    var_11 = 'test'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 'pre_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py'
    var_18 = 'import sys; sys.exit(1)'
    var_19 = 'test'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = 'pre_gen_project'
    var_23 = False
    var_24 = 'test'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = 'pre_gen_project'
    var_28 = True



# Parsed testcases at query #45
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/fake/project_dir'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No %s hook found'
    var_5 = 'pre_gen_project'
    var_6 = '/fake/project_dir'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = module_0.run_hook(var_5, var_6, var_9)
    var_11 = '/fake/hook_script.py'
    var_12 = {var_7: var_8}
    var_13 = 'post_gen_project'
    var_14 = '/fake/project_dir'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = module_0.run_hook(var_13, var_14, var_17)
    var_19 = '/fake/hook_script1.py'
    var_20 = {var_15: var_16}
    var_21 = '/fake/hook_script2.py'
    var_22 = {var_15: var_16}



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = "#!/bin/sh\necho 'Hello'\n"
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = 'post_gen_project_fail.sh'
    var_11 = '#!/bin/sh\nexit 1\n'
    var_12 = 'post_gen_project'
    var_13 = True
    var_14 = 'post_gen_project'
    var_15 = False



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'script.py'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #49
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '#!/bin/sh\necho "Hello, World!"'
    var_3 = '/nonexistent/script.py'
    var_4 = module_0.run_script(var_3)



# Parsed testcases at query #50
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
    var_10 = 'post_gen_project'
    var_11 = 'Test error'
    var_12 = 'post_gen_project'
    var_13 = False
    var_14 = 'post_gen_project'



# Parsed testcases at query #51
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'non_existent_dir'
    assert var_1 is None
    var_2 = module_0.find_hook(var_0, var_1)
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'pre_gen_project'



# Parsed testcases at query #52
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 493
    var_3 = 'test_script.sh'
    var_4 = '#!/bin/sh\nexit 1\n'
    var_5 = 493
    var_6 = 'test_script.py'
    var_7 = 'print("Hello, World!")\n'
    var_8 = 'test_script.sh'
    var_9 = 'echo "Hello, World!"\n'
    var_10 = str(var_5)
    var_11 = '/nonexistent/script.sh'
    var_12 = '.'
    var_13 = module_0.run_script(var_11, var_12)
    var_14 = str(var_11)



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'non_existent_hook'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None
    var_7 = 'cookiecutter.hooks.run_script_with_context'



# Parsed testcases at query #54
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
    var_7 = 'invalid_hook.py'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'pre_gen_project.py~'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py'
    var_14 = 'pre_gen_project.sh'
    var_15 = 'pre_gen_project'
    var_16 = len(var_5)
    assert var_16 == 2



# Parsed testcases at query #55
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
    var_5 = '# invalid hook'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = '# valid hook'
    var_9 = 'pre_gen_project'
    var_10 = 0
    var_11 = 'pre_gen_project.py'
    var_12 = 'hooks'
    var_13 = '# valid hook 1'
    var_14 = '# valid hook 2'
    var_15 = 'pre_gen_project'
    var_16 = 'pre_gen_project.py'
    var_17 = 'pre_gen_project.sh'
    var_18 = 'hooks'
    var_19 = '# backup hook'
    var_20 = 'pre_gen_project'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #58
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
    var_5 = '#!/usr/bin/env python\nprint("test")'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = '#!/usr/bin/env python\nprint("test")'
    var_9 = 'pre_gen_project'
    var_10 = 'pre_gen_project.py'
    var_11 = 'hooks'
    var_12 = '#!/usr/bin/env python\nprint("test")'
    var_13 = '#!/bin/sh\necho "test"'
    var_14 = 'pre_gen_project'
    var_15 = 'pre_gen_project.py'
    var_16 = 'pre_gen_project.sh'
    var_17 = 'hooks'
    var_18 = '#!/usr/bin/env python\nprint("test")'
    var_19 = 'pre_gen_project'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #60
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = None
    var_4 = '/fake/repo'
    var_5 = module_0.run_pre_prompt_hook(var_4)
    assert var_5 == '/tmp/fake_repo'
    var_6 = '/tmp/fake_repo'
    var_7 = '/fake/repo/hooks/pre_prompt.py'
    var_8 = None
    var_9 = 'Hook failed'
    var_10 = '/fake/repo'
    var_11 = module_0.run_pre_prompt_hook(var_10)
    var_12 = str(var_6)



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.py'
    var_4 = "print('Hook executed')"
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'post_gen_project'
    var_9 = False
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = 'hooks'
    var_13 = 'post_gen_project.py'
    var_14 = 'import sys; sys.exit(1)'
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'post_gen_project'
    var_19 = True
    var_20 = 'repo'
    var_21 = var_18 / var_20
    var_22 = 'project'
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'post_gen_project.py'
    var_26 = var_24 / var_25
    var_27 = 'import sys; sys.exit(1)'
    var_28 = 'test'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'post_gen_project'
    var_32 = False
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'nonexistent_hook'
    var_40 = False
    var_41 = 'repo'
    var_42 = var_31 / var_41
    var_43 = 'project'
    var_44 = 'hooks'
    var_45 = var_42 / var_44
    var_46 = 'post_gen_project.py'
    var_47 = var_45 / var_46
    var_48 = '{{ undefined_variable }}'
    var_49 = 'test'
    var_50 = 'value'
    var_51 = {var_49: var_50}
    var_52 = 'post_gen_project'
    var_53 = False



# Parsed testcases at query #62
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



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



# Parsed testcases at query #64
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = '/tmp/hook_script.py'
    var_7 = 'pre_gen_project'
    var_8 = '/tmp/project'
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = module_0.run_hook(var_7, var_8, var_11)
    var_13 = {var_9: var_10}
    var_14 = '/tmp/hook1.py'
    var_15 = '/tmp/hook2.sh'
    var_16 = 'post_gen_project'
    var_17 = '/tmp/project'
    var_18 = 'name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_hook(var_16, var_17, var_20)
    var_22 = {var_18: var_19}
    var_23 = {var_18: var_19}



# Parsed testcases at query #65
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'hook_script.py'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)
    assert var_8 == 'Pre-Prompt Hook script failed'



# Parsed testcases at query #66
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'dummy_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'dummy_repo_dir'
    var_2 = 'dummy_repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo_dir'
    var_4 = 'dummy_script.py'
    var_5 = 'temp_repo_dir'
    var_6 = 'dummy_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter.hooks.work_in'
    var_2 = 'cookiecutter.hooks.find_hook'
    var_3 = None
    var_4 = 'test_repo_with_hook'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.py'
    var_7 = "print('Hello, World!')"
    var_8 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = 'test_repo_with_failing_hook'
    var_11 = 'exit(1)'
    var_12 = 'Hook script failed'



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'hooks'
    var_6 = 'pre_gen_project.py'
    var_7 = 'pre_gen_project.sh'
    var_8 = 'pre_gen_project'
    var_9 = 'hooks'
    var_10 = 'invalid_hook.py'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'nonexistent_hooks_dir'
    var_14 = 'pre_gen_project'



# Parsed testcases at query #69
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
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



# Parsed testcases at query #70
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'fake_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'fake_repo_dir'
    var_2 = 'fake_repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo_dir'
    var_4 = 'fake_script'
    var_5 = 'temp_repo_dir'
    var_6 = 'fake_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #71
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = '#!/usr/bin/env python\nprint("Hello")'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = '#!/usr/bin/env python\nprint("Hello")'
    var_8 = 'pre_gen_project'
    var_9 = 'pre_gen_project.py'
    var_10 = 'hooks'
    var_11 = '#!/usr/bin/env python\nprint("Hello")'
    var_12 = '#!/bin/sh\necho "Hello"'
    var_13 = 'pre_gen_project'
    var_14 = len(var_2)
    assert var_14 == 2
    var_15 = 'pre_gen_project.py'
    var_16 = 'pre_gen_project.sh'
    var_17 = 'hooks'
    var_18 = '#!/usr/bin/env python\nprint("Hello")'
    var_19 = 'pre_gen_project'



# Parsed testcases at query #72
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo'
    var_2 = 'pre_prompt'
    var_3 = '/fake/repo/hooks/pre_prompt.py'
    var_4 = '/fake/repo'
    var_5 = module_0.run_pre_prompt_hook(var_4)
    assert var_5 == '/tmp/repo'
    var_6 = '/tmp/repo'
    var_7 = '/fake/repo/hooks/pre_prompt.py'
    var_8 = 'test error'
    var_9 = '/fake/repo'
    var_10 = module_0.run_pre_prompt_hook(var_9)
    var_11 = str(var_6)



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\nexit 1\n'
    var_4 = 'test_script.sh'
    var_5 = 'echo "Hello, World!"\n'
    var_6 = 'test_script.py'
    var_7 = 'print("Hello, World!")\n'



# Parsed testcases at query #74
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'test_repo'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo'
    var_4 = 'script.sh'
    var_5 = 'temp_repo'
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    var_8 = str(var_6)



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py~'
    var_8 = '#!/usr/bin/env python\nprint("test")'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py'
    var_12 = '#!/usr/bin/env python\nprint("test1")'
    var_13 = 'pre_gen_project.sh'
    var_14 = '#!/bin/sh\necho "test2"'
    var_15 = 'pre_gen_project'
    var_16 = 'nonexistent'
    var_17 = 'pre_gen_project'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'echo "Hello {{ project_name }}"'
    var_4 = '.sh'
    var_5 = 'utf-8'



# Parsed testcases at query #77
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



# Parsed testcases at query #78
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



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #80
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo'
    var_2 = 'pre_prompt_script'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = None
    var_6 = 'test_repo'
    var_7 = module_0.run_pre_prompt_hook(var_6)
    assert var_7 == 'temp_repo'
    var_8 = 'temp_repo'
    var_9 = 'pre_prompt_script'
    var_10 = [var_9]
    var_11 = [var_9]
    var_12 = None
    var_13 = 'Test error'
    var_14 = 'test_repo'
    var_15 = module_0.run_pre_prompt_hook(var_14)



# Parsed testcases at query #81
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo_dir'
    var_2 = 'test_script'
    var_3 = 'test_repo_dir'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == 'temp_repo_dir'
    var_5 = 'temp_repo_dir'
    var_6 = 'test_script'
    var_7 = 'Test error'
    var_8 = 'test_repo_dir'
    var_9 = module_0.run_pre_prompt_hook(var_8)



# Parsed testcases at query #82
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = '#!/bin/sh\necho "Hello, World!"\n'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = '/non/existent/script.py'
    var_4 = module_0.run_script(var_3)
    var_5 = ''



# Parsed testcases at query #83
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



# Parsed testcases at query #84
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/fake/repo'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == '/tmp/fake_repo'
    var_5 = '/tmp/fake_repo'
    var_6 = '/fake/repo/hooks/pre_prompt.py'
    var_7 = 'Script failed'
    var_8 = '/fake/repo'
    var_9 = module_0.run_pre_prompt_hook(var_8)



# Parsed testcases at query #85
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = 'name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = '/tmp/hook.py'
    var_13 = {var_8: var_9}
    var_14 = 'post_gen_project'
    var_15 = '/tmp/project'
    var_16 = 'name'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = module_0.run_hook(var_14, var_15, var_18)
    var_20 = '/tmp/hook1.py'
    var_21 = {var_16: var_17}
    var_22 = '/tmp/hook2.sh'
    var_23 = {var_16: var_17}



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'hooks'
    var_4 = 'post_gen_project.py'
    var_5 = 'print("Hook executed")'
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = 'import sys; sys.exit(1)'
    var_9 = 'post_gen_project'
    var_10 = True
    var_11 = 'post_gen_project'
    var_12 = False



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project'
    var_2 = '#!/bin/sh\necho "Hook executed"'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'hooks'
    var_8 = 'post_gen_project'
    var_9 = '#!/bin/sh\nexit 1'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'hooks'
    var_16 = 'post_gen_project'
    var_17 = '#!/bin/sh\nexit 1'
    var_18 = 'project_name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = 'post_gen_project'
    var_22 = False
    var_23 = 'project_name'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = 'post_gen_project'
    var_27 = True



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'test_script.sh'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\nexit 1'
    var_4 = 'test_script.py'
    var_5 = 'exit(0)'
    var_6 = 'nonexistent.sh'



# Parsed testcases at query #90
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'test_repo_dir'
    var_2 = 'test_script.py'
    var_3 = 'test_repo_dir'
    var_4 = module_0.run_pre_prompt_hook(var_3)
    assert var_4 == 'temp_repo_dir'
    var_5 = 'temp_repo_dir'
    var_6 = 'test_script.py'
    var_7 = 'Test error'
    var_8 = 'test_repo_dir'
    var_9 = module_0.run_pre_prompt_hook(var_8)
    var_10 = str(var_5)



# Parsed testcases at query #91
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project'
    var_5 = 'pre_gen_project.py'
    var_6 = '#!/usr/bin/env python\nprint("Hello")'
    var_7 = 'invalid_hook.py'
    var_8 = '#!/usr/bin/env python\nprint("Hello")'
    var_9 = 'pre_gen_project.py~'
    var_10 = '#!/usr/bin/env python\nprint("Hello")'
    var_11 = 'pre_gen_project.sh'
    var_12 = '#!/bin/sh\necho "Hello"'



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.sh'
    var_2 = '#!/bin/sh\necho "Success"\n'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = True
    var_8 = 'hooks'
    var_9 = 'post_gen_project.sh'
    var_10 = '#!/bin/sh\nexit 1\n'
    var_11 = 'project_name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'hooks'
    var_17 = 'post_gen_project.sh'
    var_18 = '#!/bin/sh\nexit 1\n'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = 'post_gen_project'
    var_23 = False
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = 'print("{{ undefined_var }}")'
    var_27 = 'project_name'
    var_28 = 'test'
    var_29 = {var_27: var_28}
    var_30 = 'post_gen_project'
    var_31 = True
    var_32 = 'project_name'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = 'post_gen_project'
    var_36 = True



# Parsed testcases at query #93
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '/non/existent/script.py'
    var_3 = module_0.run_script(var_2)
    var_4 = ''



# Parsed testcases at query #94
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
    var_14 = 'print("Hook 1 executed")'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'echo "Hook 2 executed"'
    var_17 = {var_4: var_5}
    var_18 = {var_4: var_5}
    var_19 = {var_4: var_5}



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'author'
    var_2 = 'test_project'
    var_3 = 'test_author'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_script.sh'
    var_6 = var_0 / var_5
    var_7 = '#!/bin/bash\necho "Project: {{ project_name }}"\necho "Author: {{ author }}"\n'
    var_8 = 'utf-8'
    var_9 = 'output.txt'
    var_10 = 'test_script.py'
    var_11 = var_0 / var_10
    var_12 = 'print("Project: {{ project_name }}")\nprint("Author: {{ author }}")\n'
    var_13 = 'utf-8'
    var_14 = 'output.txt'
    var_15 = 'test_script.sh'
    var_16 = var_0 / var_15
    var_17 = '#!/bin/bash\nexit 1\n'
    var_18 = 'utf-8'



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 1
    var_2 = '.py'
    var_3 = tempfile.mkstemp(suffix=var_2)[var_1]
    var_4 = 'utf-8'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}



# Parsed testcases at query #97
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == '/fake/tmp_repo'
    var_4 = '/fake/script.py'
    var_5 = '/fake/tmp_repo'
    var_6 = '/fake/repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)



# Parsed testcases at query #98
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
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = '{{ undefined_var }}'
    var_27 = 'project_name'
    var_28 = 'test'
    var_29 = {var_27: var_28}
    var_30 = 'post_gen_project'
    var_31 = True



# Parsed testcases at query #99
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.py'
    var_7 = ''
    var_8 = 'non_existent_script.py'



# Parsed testcases at query #100
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_prompt'
    var_1 = 'nonexistent_dir'
    var_2 = module_0.find_hook(var_0, var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_prompt'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.py'
    var_7 = 'pre_prompt'
    var_8 = 'hooks'
    var_9 = 'pre_prompt.py'
    var_10 = 'pre_prompt.sh'
    var_11 = 'pre_prompt'
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py~'
    var_14 = 'pre_prompt'
    var_15 = 'hooks'
    var_16 = 'invalid_hook.py'
    var_17 = 'pre_prompt'



# Parsed testcases at query #101
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'debug'
    var_3 = 'pre_gen_project'
    var_4 = {}
    var_5 = 'No %s hook found'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("Hook executed")'
    var_9 = 'run_script_with_context'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = {var_10: var_11}
    var_14 = 'pre_gen_project.sh'
    var_15 = 'print("Hook 1 executed")'
    var_16 = 'echo "Hook 2 executed"'
    var_17 = {var_10: var_11}



# Parsed testcases at query #102
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'test_hook'
    var_1 = '/fake/project_dir'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.run_hook(var_0, var_1, var_4)
    var_6 = 'No %s hook found'
    var_7 = '/fake/hook_script.sh'
    var_8 = 'test_hook'
    var_9 = '/fake/project_dir'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.run_hook(var_8, var_9, var_12)
    var_14 = '/fake/hook_script.sh'
    var_15 = {var_10: var_11}
    var_16 = 'Running hook %s'
    var_17 = '/fake/hook_script1.sh'
    var_18 = '/fake/hook_script2.sh'
    var_19 = 'test_hook'
    var_20 = '/fake/project_dir'
    var_21 = 'key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = module_0.run_hook(var_19, var_20, var_23)
    var_25 = '/fake/hook_script1.sh'
    var_26 = {var_21: var_22}
    var_27 = '/fake/hook_script2.sh'
    var_28 = {var_21: var_22}
    var_29 = 'Running hook %s'



# Parsed testcases at query #103
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = '/path/to/script.py'
    var_2 = [var_1]
    var_3 = 'cookiecutter.hooks.run_script_with_context'
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = None
    var_2 = 'cookiecutter.hooks.logger'
    var_3 = 'project'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'pre_gen_project'
    var_8 = 'No %s hook found'

def test_case_0():
    var_0 = 'cookiecutter.hooks.find_hook'
    var_1 = '/path/to/script1.py'
    var_2 = '/path/to/script2.py'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter.hooks.run_script_with_context'
    var_5 = 'project'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = 'pre_gen_project'



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'



# Parsed testcases at query #105
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 1
    var_2 = '.py'
    var_3 = tempfile.mkstemp(suffix=var_2)[var_1]
    var_4 = 'utf-8'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_prompt'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/fake/repo'
    var_4 = 'pre_prompt'
    var_5 = '/fake/tmp/repo'
    var_6 = '/fake/repo/hooks/pre_prompt.py'
    var_7 = 'Script failed'
    var_8 = '/fake/repo'
    var_9 = 'pre_prompt'
    var_10 = '/fake/tmp/repo'



# Parsed testcases at query #107
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'non_existent_script.py'
    var_7 = 'empty_script.py'
    var_8 = ''



# Parsed testcases at query #108
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



# Parsed testcases at query #109
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #110
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
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'invalid_hook.py'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = 'pre_gen_project'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = 'pre_gen_project.sh'
    var_16 = 'pre_gen_project'



# Parsed testcases at query #111
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



# Parsed testcases at query #112
#--------------------------


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'pre_prompt'
    var_2 = '/fake/repo/hooks/pre_prompt.py'
    var_3 = '/tmp/fake_repo'
    var_4 = None
    var_5 = '/fake/repo'
    var_6 = 'pre_prompt'
    var_7 = '/fake/repo/hooks/pre_prompt.py'
    var_8 = '/tmp/fake_repo'
    var_9 = None
    var_10 = 'Test error'
    var_11 = '/fake/repo'
    var_12 = str(var_6)
    assert var_12 == 'Pre-Prompt Hook script failed'



# Parsed testcases at query #113
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.py'
    var_5 = 'utf-8'



# Parsed testcases at query #114
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #115
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #116
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'dummy_repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == 'dummy_repo_dir'
    var_2 = 'dummy_repo_dir'
    var_3 = module_0.run_pre_prompt_hook(var_2)
    assert var_3 == 'temp_repo_dir'
    var_4 = 'dummy_script.py'
    var_5 = 'temp_repo_dir'
    var_6 = 'dummy_repo_dir'
    var_7 = module_0.run_pre_prompt_hook(var_6)



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = 'name'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = '.py'
    var_5 = '.'



# Parsed testcases at query #118
#--------------------------


def test_case_0():
    var_0 = "print('Hello, {{ name }}!')"
    var_1 = '.py'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}



# Parsed testcases at query #119
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'print("Hello, World!")'
    var_1 = 'import sys; sys.exit(1)'
    var_2 = '/nonexistent/script.py'
    var_3 = module_0.run_script(var_2)
    var_4 = 'echo "Hello, World!"'



# Parsed testcases at query #120
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/repo_dir'
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert var_1 == '/fake/repo_dir'
    var_2 = '/fake/script.py'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = '/fake/repo_dir'
    var_6 = module_0.run_pre_prompt_hook(var_5)
    assert var_6 == '/fake/tmp_repo_dir'
    var_7 = '/fake/tmp_repo_dir'
    var_8 = '/fake/script.py'
    var_9 = [var_8]
    var_10 = [var_8]
    var_11 = 'Test error'
    var_12 = '/fake/repo_dir'
    var_13 = module_0.run_pre_prompt_hook(var_12)



# Parsed testcases at query #121
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = 'pre_gen_project'
    var_3 = module_0.find_hook(var_2)
    var_4 = 'hooks'
    var_5 = 'non_existent_hook'
    var_6 = module_0.find_hook(var_5)
    assert var_6 is None
    var_7 = 'pre_gen_project'
    var_8 = module_0.find_hook(var_7)
    assert var_8 is None
    var_9 = 'hooks'
    var_10 = 'invalid_hook.py'
    var_11 = 'pre_gen_project'
    var_12 = module_0.find_hook(var_11)
    assert var_12 is None
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py~'
    var_15 = 'pre_gen_project'
    var_16 = module_0.find_hook(var_15)
    assert var_16 is None



# Parsed testcases at query #122
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("test")'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py~'
    var_8 = '#!/usr/bin/env python\nprint("test")'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'invalid_hook.py'
    var_12 = '#!/usr/bin/env python\nprint("test")'
    var_13 = 'pre_gen_project'
    var_14 = 'nonexistent'
    var_15 = 'pre_gen_project'



# Parsed testcases at query #123
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, {{ name }}!")'
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}
    var_5 = 'test_script.sh'
    var_6 = 'echo "Hello, {{ name }}!"'
    var_7 = 'name'
    var_8 = 'World'
    var_9 = {var_7: var_8}
    var_10 = 'test_script.py'
    var_11 = 'import sys; sys.exit(1)'
    var_12 = {}
    var_13 = 'test_script.py'
    var_14 = 'print("Hello, {{ undefined_var }}!")'
    var_15 = {}



# Parsed testcases at query #124
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_gen_project.py'
    var_2 = '#!/usr/bin/env python\nprint("Valid hook")'
    var_3 = 'pre_gen_project'
    var_4 = 'non_existent_hooks_dir'
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = 'invalid_hook.py'
    var_8 = '#!/usr/bin/env python\nprint("Invalid hook")'
    var_9 = 'pre_gen_project'
    var_10 = 'hooks'
    var_11 = 'pre_gen_project.py~'
    var_12 = '#!/usr/bin/env python\nprint("Backup hook")'
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py'
    var_16 = '#!/usr/bin/env python\nprint("Valid hook 1")'
    var_17 = 'pre_gen_project.sh'
    var_18 = '#!/bin/sh\necho "Valid hook 2"'
    var_19 = 'pre_gen_project'



# Parsed testcases at query #125
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = 'print("Hello, World!")'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/sh\necho "Hello, World!"'
    var_4 = 'test_script.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'test_script.sh'
    var_7 = 'echo "Hello, World!"'
    var_8 = 'test_script.py'
    var_9 = 'import os\nos.remove("nonexistent_file")'



# Parsed testcases at query #126
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'project'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter.hooks.find_hook'
    var_6 = None



