####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = 'hooks'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'post_gen_project.sh'
    var_10 = 'other_script.py'
    var_11 = 'pre_gen_project'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py~'
    var_14 = 'pre_gen_project'
    var_15 = 'hooks'
    var_16 = 'unsupported_hook.py'
    var_17 = 'unsupported_hook'
    var_18 = 'hooks'
    var_19 = 'pre_gen_project.sh'
    var_20 = 'pre_gen_project'
    var_21 = 'hooks'
    var_22 = 'pre_gen_project.py'
    var_23 = 'pre_gen_project.sh'
    var_24 = 'pre_gen_project'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "print('pre_prompt hook executed')"
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'import sys; sys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = "print('hook1')"
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'hook2'"
    var_11 = 493
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py'
    var_14 = "print('valid')"
    var_15 = 'pre_gen_project.py'
    var_16 = "print('invalid')"
    var_17 = 'pre_prompt.py~'
    var_18 = "print('backup')"



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = ''
    var_4 = 'import sys\nsys.exit(0)'
    var_5 = 292
    var_6 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('pre_prompt hook executed')"
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = "#!/usr/bin/env python\nprint('hook1')"
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'hook2'"
    var_11 = 493
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py'
    var_14 = "#!/usr/bin/env python\nprint('valid')"
    var_15 = 'pre_prompt.py~'
    var_16 = 'backup'
    var_17 = 'post_gen_project.py'
    var_18 = 'wrong hook'
    var_19 = 'hooks'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No pre_gen_project hook found'
    var_5 = '/tmp/hooks/pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = '/tmp/project'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.run_hook(var_6, var_7, var_10)
    var_12 = '/tmp/hooks/pre_gen_project.py'
    var_13 = {var_8: var_9}
    var_14 = '/tmp/hooks/pre_gen_project.py'
    var_15 = '/tmp/hooks/pre_gen_project.sh'
    var_16 = 'pre_gen_project'
    var_17 = '/tmp/project'
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_hook(var_16, var_17, var_20)
    var_22 = '/tmp/hooks/pre_gen_project.py'
    var_23 = {var_18: var_19}
    var_24 = '/tmp/hooks/pre_gen_project.sh'
    var_25 = {var_18: var_19}
    var_26 = True
    var_27 = '/tmp/hooks/pre_gen_project.py'
    var_28 = 'Hook failed'
    var_29 = 'pre_gen_project'
    var_30 = '/tmp/project'
    var_31 = {}
    var_32 = module_0.run_hook(var_29, var_30, var_31)
    var_33 = '/tmp/hooks/pre_gen_project.py'
    var_34 = 'Template error'
    var_35 = 'pre_gen_project'
    var_36 = '/tmp/project'
    var_37 = {}
    var_38 = module_0.run_hook(var_35, var_36, var_37)
    var_39 = '/tmp/hooks/pre_gen_project.py'
    var_40 = 'pre_gen_project'
    var_41 = '/tmp/project'
    var_42 = {}
    var_43 = module_0.run_hook(var_40, var_41, var_42)
    var_44 = 'Running hook pre_gen_project'



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = '#!/bin/sh\nexit 1'
    var_4 = 'test_script.py'
    var_5 = var_3 / var_4
    var_6 = 'import sys\nsys.exit(0)'
    var_7 = str(var_5)
    var_8 = 'import sys\nsys.exit(0)'
    var_9 = var_8.st_mode
    var_10 = var_6.st_mode



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = 'hooks'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'pre_gen_project.sh'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py'
    var_13 = 'pre_gen_project.py~'
    var_14 = 'pre_gen_project'
    var_15 = 'hooks'
    var_16 = 'unsupported_hook.py'
    var_17 = 'unsupported_hook'
    var_18 = 'hooks'
    var_19 = 'post_gen_project.py'
    var_20 = 'pre_gen_project'
    var_21 = 'hooks'
    var_22 = 'pre_prompt'
    var_23 = [var_20]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = 'hooks'
    var_3 = 'pre_gen_project'
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'post_gen_project.py'
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = 'pre_gen_project.py~'
    var_13 = 'pre_gen_project'
    var_14 = 'hooks'
    var_15 = 'unsupported_hook.py'
    var_16 = 'unsupported_hook'
    var_17 = 'hooks'
    var_18 = 'pre_gen_project'
    var_19 = [var_16]
    var_20 = 'hooks'
    var_21 = 'pre_gen_project.sh'
    var_22 = 'pre_gen_project'
    var_23 = [var_19]
    var_24 = 'hooks'
    var_25 = 'post_gen_project.py'
    var_26 = 'pre_gen_project'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'pre_gen_project.py'
    var_4 = "print('test')"
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = var_8 / var_10
    var_12 = 'project'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = 'Hook failed'
    var_17 = 'pre_gen_project'
    var_18 = True
    var_19 = 'repo'
    var_20 = var_17 / var_19
    var_21 = 'project'
    var_22 = 'key'
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = 'Template error'
    var_26 = 'pre_gen_project'
    var_27 = True
    var_28 = 'repo'
    var_29 = var_26 / var_28
    var_30 = 'project'
    var_31 = 'key'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = 'Hook failed'
    var_35 = 'pre_gen_project'
    var_36 = False
    var_37 = 'repo'
    var_38 = var_35 / var_37
    var_39 = 'project'
    var_40 = 'key'
    var_41 = 'value'
    var_42 = {var_40: var_41}
    var_43 = 'pre_gen_project'
    var_44 = True



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("Hello {{ name }}!")\nprint("Project: {{ project_name }}")\nsys.exit(0)\n'
    var_1 = 'test_script.py'
    var_2 = 'name'
    var_3 = 'project_name'
    var_4 = 'TestUser'
    var_5 = 'TestProject'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'work_dir'

def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("{{ greeting }}")\nsys.exit(1)\n'
    var_1 = 'failing_script.py'
    var_2 = 'greeting'
    var_3 = 'This will fail'
    var_4 = {var_2: var_3}
    var_5 = 'work_dir'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = ''
    var_2 = {}
    var_3 = 'work_dir'
    var_4 = module_0.run_script_with_context(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'This is not a valid script\n{{ variable }}\nJust some text\n'
    var_1 = 'invalid_script.txt'
    var_2 = 'variable'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'work_dir'

def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("{{ undefined_variable }}")\nsys.exit(0)\n'
    var_1 = 'template_error_script.py'
    var_2 = 'defined_variable'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'work_dir'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'binary_script.py'
    var_1 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'
    var_2 = {}
    var_3 = 'work_dir'
    var_4 = module_0.run_script_with_context(var_0, var_1, var_2)

def test_case_0():
    var_0 = '@echo off\necho Hello {{ name }}!\n'
    var_1 = 'windows_script.bat'
    var_2 = 'name'
    var_3 = 'WindowsUser'
    var_4 = {var_2: var_3}
    var_5 = 'work_dir'
    var_6 = 'platform'
    var_7 = 'win32'

def test_case_0():
    var_0 = '#!/bin/bash\necho "Project: {{ project_name }}"\nexit 0\n'
    var_1 = 'shell_script.sh'
    var_2 = 'project_name'
    var_3 = 'ShellProject'
    var_4 = {var_2: var_3}
    var_5 = 'work_dir'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/some/dir'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No pre_gen_project hook found'
    var_5 = '/hooks/pre_gen_project.py'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 'pre_gen_project'
    var_10 = '/project/dir'
    var_11 = module_0.run_hook(var_9, var_10, var_8)
    var_12 = 'Running hook pre_gen_project'
    var_13 = '/hooks/pre_gen_project.py'
    var_14 = '/hooks/pre_gen_project.py'
    var_15 = '/hooks/pre_gen_project.sh'
    var_16 = 'project_name'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = 'pre_gen_project'
    var_20 = '/project/dir'
    var_21 = module_0.run_hook(var_19, var_20, var_18)
    var_22 = 'Running hook pre_gen_project'
    var_23 = '/hooks/pre_gen_project.py'
    var_24 = '/hooks/pre_gen_project.sh'
    var_25 = '/hooks/invalid_hook.py'
    var_26 = 'invalid_hook'
    var_27 = '/project/dir'
    var_28 = {}
    var_29 = module_0.run_hook(var_26, var_27, var_28)
    var_30 = 'pre_gen_project'
    var_31 = '/some/dir'
    var_32 = {}
    var_33 = module_0.run_hook(var_30, var_31, var_32)
    var_34 = 'No pre_gen_project hook found'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('pre_prompt executed')"
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = "#!/usr/bin/env python\nprint('hook1')"
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'hook2'"
    var_11 = 493
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py'
    var_14 = "#!/usr/bin/env python\nprint('valid')"
    var_15 = 'pre_prompt.py~'
    var_16 = 'backup'
    var_17 = 'post_gen_project.py'
    var_18 = 'wrong name'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello {{ name }}')"
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = {var_2: var_3}
    var_7 = 'test_script.sh'
    var_8 = var_2 / var_7
    var_9 = "echo 'Hello {{ name }}'"
    var_10 = 'name'
    var_11 = 'World'
    var_12 = {var_10: var_11}
    var_13 = 'test_script.py'
    var_14 = var_10 / var_13
    var_15 = "print('Project: {{ project_name }}, Version: {{ version }}')"
    var_16 = 'project_name'
    var_17 = 'version'
    var_18 = 'author'
    var_19 = 'MyProject'
    var_20 = '1.0'
    var_21 = 'Test Author'
    var_22 = {var_16: var_19, var_17: var_20, var_18: var_21}
    var_23 = 'test_script.py'
    var_24 = var_16 / var_23
    var_25 = "print('test')"
    var_26 = {}
    var_27 = 0
    var_28 = '.py'
    var_29 = 'empty_script.py'
    var_30 = var_26 / var_29
    var_31 = ''
    var_32 = {}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = True
    var_7 = 'repo'
    var_8 = var_5 / var_7
    var_9 = 'project'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 'pre_gen_project'
    var_14 = True
    var_15 = 'repo'
    var_16 = var_13 / var_15
    var_17 = 'project'
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'Hook failed'
    var_22 = 'pre_gen_project'
    var_23 = True
    var_24 = 'repo'
    var_25 = var_22 / var_24
    var_26 = 'project'
    var_27 = 'key'
    var_28 = 'value'
    var_29 = {var_27: var_28}
    var_30 = 'Template error'
    var_31 = 'pre_gen_project'
    var_32 = True
    var_33 = 'repo'
    var_34 = var_31 / var_33
    var_35 = 'project'
    var_36 = 'key'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = 'Hook failed'
    var_40 = 'pre_gen_project'
    var_41 = False
    var_42 = 'repo'
    var_43 = var_40 / var_42
    var_44 = 'project'
    var_45 = 'key'
    var_46 = 'value'
    var_47 = {var_45: var_46}
    var_48 = None
    var_49 = 'post_gen_project'
    var_50 = True



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('{{ greeting }} {{ name }}!')"
    var_2 = 'utf-8'
    var_3 = 'greeting'
    var_4 = 'name'
    var_5 = 'Hello'
    var_6 = 'World'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None
    var_9 = 1
    var_10 = var_8[var_9]
    var_11 = '.py'
    var_12 = 'test_script.sh'
    var_13 = "#!/bin/bash\necho '{{ message }}'"
    var_14 = 'utf-8'
    var_15 = 'message'
    var_16 = 'Test message'
    var_17 = {var_15: var_16}
    var_18 = None
    var_19 = 0
    var_20 = var_18[var_19]
    var_21 = '.sh'
    var_22 = 'test_script.py'
    var_23 = "print('{{ variable }}')"
    var_24 = 'utf-8'
    var_25 = 'variable'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = 'test_script.py'
    var_29 = "print('{{ undefined_variable }}')"
    var_30 = 'utf-8'
    var_31 = 'defined_variable'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = 'test_script.py'
    var_35 = "print('Static content')"
    var_36 = 'utf-8'
    var_37 = {}



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("pre_prompt hook executed")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'import sys\nsys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt'
    var_8 = '#!/bin/bash\necho "test"'
    var_9 = 493
    var_10 = 'hooks'
    var_11 = 'pre_prompt.py'
    var_12 = 'print("hook1")'
    var_13 = 'pre_prompt.sh'
    var_14 = '#!/bin/bash\necho "hook2"'
    var_15 = 493
    var_16 = 'hooks'
    var_17 = 'pre_prompt.py'
    var_18 = 'print("valid")'
    var_19 = 'pre_prompt.py~'
    var_20 = 'print("backup")'
    var_21 = 'wrong_name.py'
    var_22 = 'print("wrong")'
    var_23 = 'unsupported_hook.py'
    var_24 = 'print("unsupported")'



# Parsed testcases at query #18
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
    var_6 = 'pre_prompt'
    var_7 = module_0.valid_hook(var_6, var_6)
    assert var_7 is True
    var_8 = 'invalid_hook.py'
    var_9 = 'invalid_hook'
    var_10 = module_0.valid_hook(var_8, var_9)
    assert var_10 is False
    var_11 = module_0.valid_hook(var_0, var_4)
    assert var_11 is False
    var_12 = 'pre_gen_project.py~'
    var_13 = module_0.valid_hook(var_12, var_1)
    assert var_13 is False
    var_14 = 'post_gen_project.sh~'
    var_15 = module_0.valid_hook(var_14, var_4)
    assert var_15 is False
    var_16 = 'unsupported_hook.py'
    var_17 = 'unsupported_hook'
    var_18 = module_0.valid_hook(var_16, var_17)
    assert var_18 is False
    var_19 = 'pre_gen_project.sh'
    var_20 = module_0.valid_hook(var_19, var_1)
    assert var_20 is True
    var_21 = module_0.valid_hook(var_1, var_1)
    assert var_21 is True
    var_22 = 'pre_gen_project.bat'
    var_23 = module_0.valid_hook(var_22, var_1)
    assert var_23 is True



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("pre_prompt hook executed")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'print("hook 1")'
    var_6 = 'pre_prompt.sh'
    var_7 = '#!/bin/bash\necho "hook 2"'
    var_8 = 493
    var_9 = 'hooks'
    var_10 = 'pre_prompt.py'
    var_11 = 'import sys\nsys.exit(1)'
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py'
    var_14 = 'print("valid")'
    var_15 = 'pre_gen_project.py'
    var_16 = 'print("should not run")'
    var_17 = 'pre_prompt.py~'
    var_18 = 'print("backup")'
    var_19 = 'hooks'



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.py'
    var_3 = 'import sys\nsys.exit(0)'
    var_4 = 'hooks'
    var_5 = 'pre_prompt.py'
    var_6 = 'import sys\nsys.exit(1)'
    var_7 = 'hooks'
    var_8 = 'pre_prompt'
    var_9 = '@echo off\nexit /b 0'
    var_10 = '#!/bin/bash\nexit 0'
    var_11 = var_6.st_mode
    var_12 = 'hooks'
    var_13 = var_10 / var_12
    var_14 = 'pre_prompt.py'
    var_15 = var_13 / var_14
    var_16 = 'import sys\nsys.exit(0)'
    var_17 = 'pre_prompt.sh'
    var_18 = var_13 / var_17
    var_19 = '@echo off\nexit /b 0'
    var_20 = '#!/bin/bash\nexit 0'
    var_21 = 'hooks'
    var_22 = var_20 / var_21
    var_23 = 'pre_prompt.py~'
    var_24 = var_22 / var_23
    var_25 = 'import sys\nsys.exit(1)'
    var_26 = 'pre_prompt.py'
    var_27 = var_22 / var_26
    var_28 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'print("pre_prompt hook executed")'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.py'
    var_7 = 'print("hook1")'
    var_8 = 'pre_prompt.sh'
    var_9 = '#!/bin/bash\necho "hook2"'
    var_10 = 'hooks'
    var_11 = 'pre_prompt.py'
    var_12 = 'import sys\nsys.exit(1)'
    var_13 = 'hooks'
    var_14 = 'pre_prompt.py'
    var_15 = '{{ cookiecutter.project_name }}'
    var_16 = 'hooks'
    var_17 = 'pre_prompt.py~'
    var_18 = 'pre_prompt.py.bak'
    var_19 = 'hooks'
    var_20 = 'invalid_hook.py'
    var_21 = 'pre_gen_project.py'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'post_gen_project.py'
    var_2 = 'hooks'
    var_3 = 'pre_prompt.py'
    var_4 = 'import sys\nsys.exit(0)'
    var_5 = 'hooks'
    var_6 = 'pre_prompt.py'
    var_7 = 'import sys\nsys.exit(1)'
    var_8 = 'hooks'
    var_9 = 'pre_prompt'
    var_10 = '@echo off\nexit /b 0'
    var_11 = '#!/bin/sh\nexit 0'
    var_12 = var_9 | var_7
    var_13 = 'hooks'
    var_14 = var_11 / var_13
    var_15 = 'pre_prompt.py'
    var_16 = var_14 / var_15
    var_17 = 'import sys\nsys.exit(0)'
    var_18 = 'pre_prompt.sh'
    var_19 = var_14 / var_18
    var_20 = '@echo off\nexit /b 0'
    var_21 = '#!/bin/sh\nexit 0'
    var_22 = var_15 | var_17
    var_23 = 'hooks'
    var_24 = var_21 / var_23
    var_25 = 'pre_prompt.py~'
    var_26 = var_24 / var_25
    var_27 = 'import sys\nsys.exit(1)'
    var_28 = 'pre_prompt.py'
    var_29 = var_24 / var_28
    var_30 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = True
    var_7 = 'repo'
    var_8 = var_5 / var_7
    var_9 = 'project'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 'pre_gen_project'
    var_14 = True
    var_15 = 'repo'
    var_16 = var_13 / var_15
    var_17 = 'project'
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'pre_gen_project'
    var_22 = True
    var_23 = 'repo'
    var_24 = var_21 / var_23
    var_25 = 'project'
    var_26 = 'key'
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = 'pre_gen_project'
    var_30 = False
    var_31 = 'repo'
    var_32 = var_29 / var_31
    var_33 = 'project'
    var_34 = 'key'
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = 'pre_gen_project'
    var_38 = True
    var_39 = 'repo'
    var_40 = var_37 / var_39
    var_41 = 'project'
    var_42 = 'key'
    var_43 = 'value'
    var_44 = {var_42: var_43}
    var_45 = 'pre_gen_project'
    var_46 = False
    var_47 = 'repo'
    var_48 = var_45 / var_47
    var_49 = 'project'
    var_50 = 'key'
    var_51 = 'value'
    var_52 = {var_50: var_51}
    var_53 = 'pre_gen_project'
    var_54 = True



# Parsed testcases at query #3
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = var_1.st_mode
    var_3 = 'import sys\nsys.exit(1)'
    var_4 = ''
    var_5 = ''
    var_6 = var_5.st_mode
    var_7 = 'test.py'
    var_8 = var_5 / var_7
    var_9 = 'import sys\nsys.exit(0)'
    var_10 = str(var_8)
    var_11 = 'import sys\nsys.exit(0)'
    var_12 = module_0.run_script(var_8)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = "print('{{ project_name }}')"
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'test_script.py'
    var_5 = var_1 / var_4
    var_6 = "import sys; print('TestProject'); sys.exit(0)"
    var_7 = "print('TestProject')"
    var_8 = 'MockProc'
    var_9 = ()
    var_10 = {}
    var_11 = type(var_8, var_9, var_10)
    var_12 = 0

def test_case_0():
    var_0 = "print('{{ project_name }}')"
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'test_script.py'
    var_5 = var_1 / var_4
    var_6 = 'import sys; sys.exit(1)'
    var_7 = 'MockProc'
    var_8 = ()
    var_9 = {}
    var_10 = type(var_7, var_8, var_9)
    var_11 = 1

def test_case_0():
    var_0 = 'Project: {{ project_name }}, Version: {{ version }}'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'MyApp'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'test_script.py'
    var_7 = var_1 / var_6
    var_8 = []
    var_9 = 'MockProc'
    var_10 = ()
    var_11 = {}
    var_12 = type(var_9, var_10, var_11)
    var_13 = 0
    var_14 = len(var_8)
    var_15 = -1
    var_16 = var_8[var_15]

def test_case_0():
    var_0 = "#!/bin/bash\necho '{{ message }}'"
    var_1 = 'message'
    var_2 = 'Hello World'
    var_3 = {var_1: var_2}
    var_4 = 'test_script.sh'
    var_5 = var_1 / var_4
    var_6 = 'MockProc'
    var_7 = ()
    var_8 = {}
    var_9 = type(var_6, var_7, var_8)
    var_10 = 0

def test_case_0():
    var_0 = "#!/bin/bash\necho '{{ message }}'"
    var_1 = 'message'
    var_2 = 'Hello World'
    var_3 = {var_1: var_2}
    var_4 = 'test_script.sh'
    var_5 = var_1 / var_4
    var_6 = 'MockProc'
    var_7 = ()
    var_8 = {}
    var_9 = type(var_6, var_7, var_8)
    var_10 = 0



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello {{ name }}!')"
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = 'wb'
    var_7 = '.py'
    var_8 = '/tmp/temp123.py'
    var_9 = 'test_script.sh'
    var_10 = "echo 'Hello {{ name }}!'"
    var_11 = 'name'
    var_12 = 'Test'
    var_13 = {var_11: var_12}
    var_14 = False
    var_15 = 'wb'
    var_16 = '.sh'
    var_17 = 'test_script.py'
    var_18 = "print('Hello {{ name }}!')"
    var_19 = 'name'
    var_20 = 'UTF-8 Test'
    var_21 = {var_19: var_20}
    var_22 = 'utf-8'
    var_23 = b"print('Hello UTF-8 Test!')"
    var_24 = 'test_script.py'
    var_25 = "print('{{ value }}')"
    var_26 = 'value'
    var_27 = 42
    var_28 = {var_26: var_27}
    var_29 = '/tmp/temp123.py'



# Parsed testcases at query #6
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
    var_6 = 'pre_prompt.py'
    var_7 = 'pre_prompt'
    var_8 = module_0.valid_hook(var_6, var_7)
    assert var_8 is True
    var_9 = 'invalid_hook.py'
    var_10 = 'invalid_hook'
    var_11 = module_0.valid_hook(var_9, var_10)
    assert var_11 is False
    var_12 = 'random_script.py'
    var_13 = 'random_script'
    var_14 = module_0.valid_hook(var_12, var_13)
    assert var_14 is False
    var_15 = 'pre_gen_project.py~'
    var_16 = module_0.valid_hook(var_15, var_1)
    assert var_16 is False
    var_17 = 'post_gen_project.sh~'
    var_18 = module_0.valid_hook(var_17, var_4)
    assert var_18 is False
    var_19 = module_0.valid_hook(var_0, var_4)
    assert var_19 is False
    var_20 = module_0.valid_hook(var_3, var_1)
    assert var_20 is False
    var_21 = module_0.valid_hook(var_1, var_1)
    assert var_21 is True
    var_22 = 'pre_gen_project.exe'
    var_23 = module_0.valid_hook(var_22, var_1)
    assert var_23 is True
    var_24 = 'unsupported_hook.py'
    var_25 = 'unsupported_hook'
    var_26 = module_0.valid_hook(var_24, var_25)
    assert var_26 is False



# Parsed testcases at query #7
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'hooks'
    var_3 = 'pre_gen_project.py'
    var_4 = "print('hook executed')"
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'
    var_9 = True
    var_10 = 'repo'
    var_11 = 'project'
    var_12 = var_9 / var_11
    var_13 = 'project_name'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = 'Hook failed'
    var_17 = 'pre_gen_project'
    var_18 = str(var_12)
    var_19 = True
    var_20 = module_0.run_hook_from_repo_dir(var_16, var_17, var_18, var_15, var_19)
    var_21 = str(var_12)
    var_22 = 'repo'
    var_23 = var_16 / var_22
    var_24 = 'project'
    var_25 = var_19 / var_24
    var_26 = 'project_name'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = 'Template error'
    var_30 = str(var_23)
    var_31 = 'pre_gen_project'
    var_32 = str(var_25)
    var_33 = False
    var_34 = module_0.run_hook_from_repo_dir(var_30, var_31, var_32, var_28, var_33)
    var_35 = 'repo'
    var_36 = var_30 / var_35
    var_37 = 'project'
    var_38 = var_33 / var_37
    var_39 = 'project_name'
    var_40 = 'test'
    var_41 = {var_39: var_40}
    var_42 = 'Hook failed'
    var_43 = str(var_36)
    var_44 = 'pre_gen_project'
    var_45 = str(var_38)
    var_46 = False
    var_47 = module_0.run_hook_from_repo_dir(var_43, var_44, var_45, var_41, var_46)
    var_48 = 'repo'
    var_49 = var_43 / var_48
    var_50 = 'project'
    var_51 = var_46 / var_50
    var_52 = 'project_name'
    var_53 = 'test'
    var_54 = {var_52: var_53}
    var_55 = str(var_49)
    var_56 = 'pre_gen_project'
    var_57 = str(var_51)
    var_58 = True
    var_59 = module_0.run_hook_from_repo_dir(var_55, var_56, var_57, var_54, var_58)
    var_60 = str(var_49)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = ''
    var_4 = '/non/existent/path/script.py'
    var_5 = module_0.run_script(var_4)
    var_6 = 'test.py'
    var_7 = var_4 / var_6
    var_8 = 'import sys\nsys.exit(0)'
    var_9 = str(var_7)
    var_10 = 'import sys\nsys.exit(0)'
    var_11 = module_0.run_script(var_7)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('{{ greeting }} {{ name }}!')"
    var_2 = 'utf-8'
    var_3 = 'greeting'
    var_4 = 'name'
    var_5 = 'Hello'
    var_6 = 'World'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None
    var_9 = 1
    var_10 = var_8[var_9]
    var_11 = '.py'
    var_12 = var_8[var_9]
    var_13 = 'utf-8'
    var_14 = 'test_script.py'
    var_15 = "print('{{ undefined_variable }}')"
    var_16 = 'utf-8'
    var_17 = 'defined_variable'
    var_18 = 'test'
    var_19 = {var_17: var_18}
    var_20 = 'test_script.sh'
    var_21 = "#!/bin/bash\necho '{{ message }}'"
    var_22 = 'utf-8'
    var_23 = 'message'
    var_24 = 'Test message'
    var_25 = {var_23: var_24}
    var_26 = None
    var_27 = 0
    var_28 = var_26[var_27]
    var_29 = '.sh'
    var_30 = 'test_script.py'
    var_31 = "print('{{ text }}')"
    var_32 = 'utf-8'
    var_33 = 'text'
    var_34 = 'Hello'
    var_35 = {var_33: var_34}
    var_36 = 'test_script.py'
    var_37 = "# -*- coding: utf-8 -*-\nprint('{{ greeting }}')"
    var_38 = 'utf-8'
    var_39 = 'greeting'
    var_40 = 'Привет'
    var_41 = {var_39: var_40}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = 'print("pre_prompt hook executed")'
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = 'import sys; sys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = 'print("hook1")'
    var_9 = 'pre_prompt.sh'
    var_10 = '#!/bin/bash\necho "hook2"'
    var_11 = 493
    var_12 = 'hooks'
    var_13 = 'pre_prompt.py'
    var_14 = 'print("valid")'
    var_15 = 'post_gen_project.py'
    var_16 = 'print("wrong hook")'
    var_17 = 'pre_prompt.py~'
    var_18 = 'print("backup")'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = ''
    var_4 = 'nonexistent_script'
    var_5 = 'test.py'
    var_6 = 'import os\nprint(os.getcwd())\nimport sys\nsys.exit(0)'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = str(var_2)
    var_4 = '/non/existent/script.py'
    var_5 = module_0.run_script(var_4)
    var_6 = str(var_4)
    var_7 = ''
    var_8 = 493
    var_9 = str(var_7)
    var_10 = 'test.py'
    var_11 = 'import os\nprint(os.getcwd())\nimport sys\nsys.exit(0)'
    var_12 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/fake/path/hook.py'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'pre_gen_project'
    var_5 = '/fake/project_dir'
    var_6 = module_0.run_hook(var_4, var_5, var_3)
    var_7 = '/fake/path/hook1.py'
    var_8 = '/fake/path/hook2.py'
    var_9 = 'post_gen_project'
    var_10 = '/another/project_dir'
    var_11 = module_0.run_hook(var_9, var_10, var_3)
    var_12 = 'pre_prompt'
    var_13 = '/empty/project_dir'
    var_14 = module_0.run_hook(var_12, var_13, var_3)
    var_15 = module_0.run_hook(var_4, var_13, var_3)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'repo'
    var_1 = 'project'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'pre_gen_project'
    var_6 = True
    var_7 = 'repo'
    var_8 = var_5 / var_7
    var_9 = 'project'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 'pre_gen_project'
    var_14 = True
    var_15 = 'repo'
    var_16 = var_13 / var_15
    var_17 = 'project'
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'pre_gen_project'
    var_22 = True
    var_23 = 'repo'
    var_24 = var_21 / var_23
    var_25 = 'project'
    var_26 = 'key'
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = 'pre_gen_project'
    var_30 = False
    var_31 = 'repo'
    var_32 = var_29 / var_31
    var_33 = 'project'
    var_34 = 'key'
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = 'pre_gen_project'
    var_38 = True
    var_39 = 'repo'
    var_40 = var_37 / var_39
    var_41 = 'project'
    var_42 = 'key'
    var_43 = 'value'
    var_44 = {var_42: var_43}
    var_45 = 'post_gen_project'
    var_46 = True



# Parsed testcases at query #15
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'import sys\nsys.exit(0)'
    var_1 = '#!/bin/sh\nexit 0'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = str(var_2)
    var_4 = ''
    var_5 = '/non/existent/script.py'
    var_6 = module_0.run_script(var_5)
    var_7 = str(var_5)
    var_8 = 'test_script.py'
    var_9 = var_5 / var_8
    var_10 = 'import sys\nsys.exit(0)'
    var_11 = str(var_9)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'author'
    var_2 = 'TestProject'
    var_3 = 'TestAuthor'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '#!/usr/bin/env python\nprint("Project: {{ project_name }}")\nprint("Author: {{ author }}")\n'
    var_6 = 'test_script.py'
    var_7 = 'utf-8'
    var_8 = 'run_dir'
    var_9 = str(var_1)
    var_10 = str(var_3)
    var_11 = str(var_7)

def test_case_0():
    var_0 = 'message'
    var_1 = 'Hello World'
    var_2 = {var_0: var_1}
    var_3 = '#!/bin/bash\necho "{{ message }}"\n'
    var_4 = 'test_script.sh'
    var_5 = 'utf-8'
    var_6 = 'run_dir'

def test_case_0():
    var_0 = 'defined_var'
    var_1 = 'I am defined'
    var_2 = {var_0: var_1}
    var_3 = '#!/usr/bin/env python\nprint("{{ defined_var }}")\nprint("{{ undefined_var }}")\n'
    var_4 = 'test_script.py'
    var_5 = 'utf-8'
    var_6 = 'run_dir'

def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = "print('test')"
    var_4 = 'test.py'
    var_5 = 'utf-8'
    var_6 = 'run_dir'
    var_7 = []
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_7[var_9]



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'hooks'
    var_1 = 'pre_prompt.py'
    var_2 = "#!/usr/bin/env python\nprint('pre_prompt hook executed')"
    var_3 = 'hooks'
    var_4 = 'pre_prompt.py'
    var_5 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_6 = 'hooks'
    var_7 = 'pre_prompt.py'
    var_8 = "#!/usr/bin/env python\nprint('hook1')"
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'hook2'"
    var_11 = 'hooks'
    var_12 = 'pre_prompt.py'
    var_13 = "#!/usr/bin/env python\nprint('valid')"
    var_14 = 'pre_prompt.py~'
    var_15 = 'backup'
    var_16 = 'post_gen_project.py'
    var_17 = 'wrong hook'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = 'hooks'
    var_2 = var_0 / var_1
    var_3 = 'pre_gen_project'
    var_4 = str(var_2)
    var_5 = module_0.find_hook(var_3, var_4)
    assert var_5 is None
    var_6 = 'hooks'
    var_7 = var_0 / var_6
    var_8 = 'pre_gen_project.py'
    var_9 = var_7 / var_8
    var_10 = 'print("test")'
    var_11 = 'pre_gen_project'
    var_12 = str(var_7)
    var_13 = module_0.find_hook(var_11, var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 0
    var_16 = var_13[var_15]
    var_17 = 'hooks'
    var_18 = var_0 / var_17
    var_19 = 'pre_gen_project.py'
    var_20 = var_18 / var_19
    var_21 = 'print("test1")'
    var_22 = 'pre_gen_project.sh'
    var_23 = var_18 / var_22
    var_24 = '#!/bin/bash\necho "test2"'
    var_25 = 'pre_gen_project'
    var_26 = str(var_18)
    var_27 = module_0.find_hook(var_25, var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = [os.path.basename(p) for p in var_27]
    var_30 = 'hooks'
    var_31 = var_0 / var_30
    var_32 = 'post_gen_project.py'
    var_33 = var_31 / var_32
    var_34 = 'print("test")'
    var_35 = 'pre_gen_project'
    var_36 = str(var_31)
    var_37 = module_0.find_hook(var_35, var_36)
    assert var_37 is None
    var_38 = 'hooks'
    var_39 = var_0 / var_38
    var_40 = 'pre_gen_project.py~'
    var_41 = var_39 / var_40
    var_42 = 'print("test")'
    var_43 = 'pre_gen_project'
    var_44 = str(var_39)
    var_45 = module_0.find_hook(var_43, var_44)
    assert var_45 is None
    var_46 = 'hooks'
    var_47 = var_0 / var_46
    var_48 = 'unsupported_hook.py'
    var_49 = var_47 / var_48
    var_50 = 'print("test")'
    var_51 = 'unsupported_hook'
    var_52 = str(var_47)
    var_53 = module_0.find_hook(var_51, var_52)
    assert var_53 is None
    var_54 = 'hooks'
    var_55 = var_46 / var_54
    var_56 = 'pre_gen_project.py'
    var_57 = var_55 / var_56
    var_58 = 'print("test")'
    var_59 = 'pre_gen_project'
    var_60 = module_0.find_hook(var_59)
    var_61 = len(var_60)
    assert var_61 == 1



# Parsed testcases at query #19
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'pre_gen_project'
    var_1 = '/tmp/project'
    var_2 = {}
    var_3 = module_0.run_hook(var_0, var_1, var_2)
    var_4 = 'No pre_gen_project hook found'
    var_5 = '/tmp/hooks/pre_gen_project.py'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 'pre_gen_project'
    var_10 = '/tmp/project'
    var_11 = module_0.run_hook(var_9, var_10, var_8)
    var_12 = 'Running hook pre_gen_project'
    var_13 = '/tmp/hooks/pre_gen_project.py'
    var_14 = '/tmp/hooks/pre_gen_project.py'
    var_15 = '/tmp/hooks/pre_gen_project.sh'
    var_16 = 'project_name'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = 'pre_gen_project'
    var_20 = '/tmp/project'
    var_21 = module_0.run_hook(var_19, var_20, var_18)
    var_22 = '/tmp/hooks/pre_gen_project.py'
    var_23 = '/tmp/hooks/pre_gen_project.sh'
    var_24 = '/tmp/hooks/pre_gen_project.py'
    var_25 = 'Hook failed'
    var_26 = 'pre_gen_project'
    var_27 = '/tmp/project'
    var_28 = {}
    var_29 = module_0.run_hook(var_26, var_27, var_28)
    var_30 = '/tmp/hooks/pre_gen_project.py'
    var_31 = 'Template error'
    var_32 = 'pre_gen_project'
    var_33 = '/tmp/project'
    var_34 = {}
    var_35 = module_0.run_hook(var_32, var_33, var_34)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("Project: {{ project_name }}")\nprint("Version: {{ version }}")\nsys.exit(0)\n'
    var_1 = 'test_script.py'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'TestProject'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'project_dir'

def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("Project: {{ project_name }}")\nsys.exit(1)\n'
    var_1 = 'test_script.py'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = 'project_dir'

def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\nprint("Project: {{ undefined_var }}")\nsys.exit(0)\n'
    var_1 = 'test_script.py'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = 'project_dir'

def test_case_0():
    var_0 = '#!/bin/bash\necho "Project: {{ project_name }}"\necho "Author: {{ author }}"\n'
    var_1 = 'test_script.sh'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'TestProject'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'project_dir'

def test_case_0():
    var_0 = 'empty_script.py'
    var_1 = ''
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = 'project_dir'

def test_case_0():
    var_0 = '#!/usr/bin/env python\nimport sys\n{% if use_database %}\nprint("Database: {{ db_name }}")\n{% else %}\nprint("No database configured")\n{% endif %}\nsys.exit(0)\n'
    var_1 = 'test_script.py'
    var_2 = 'use_database'
    var_3 = 'db_name'
    var_4 = True
    var_5 = 'test_db'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'project_dir'
    var_8 = False
    var_9 = {var_2: var_8, var_3: var_5}



