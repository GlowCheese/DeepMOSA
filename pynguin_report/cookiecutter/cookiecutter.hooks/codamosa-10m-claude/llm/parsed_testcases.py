####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = 'test_script.py'
    var_2 = "\nimport os\nwith open('{{ output_file }}', 'w') as f:\n    f.write('{{ greeting }} {{ name }}')\n"
    var_3 = 'utf-8'
    var_4 = 'output_file'
    var_5 = 'greeting'
    var_6 = 'name'
    var_7 = 'output.txt'
    var_8 = 'Hello'
    var_9 = 'World'

def test_case_0():
    var_0 = 'Test run_script_with_context with bash script.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho "{{ message }}" > {{ output_file }}\n'
    var_3 = 'utf-8'
    var_4 = 'message'
    var_5 = 'output_file'
    var_6 = 'Test message'
    var_7 = 'output.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context raises UndefinedError for missing context variables.'
    var_1 = 'test_script.py'
    var_2 = "\nprint('{{ undefined_var }}')\n"
    var_3 = 'utf-8'
    var_4 = {}

def test_case_0():
    var_0 = 'Test run_script_with_context executes in specified working directory.'
    var_1 = 'test_script.py'
    var_2 = "\nimport os\nwith open('result.txt', 'w') as f:\n    f.write(os.getcwd())\n"
    var_3 = 'utf-8'
    var_4 = 'workdir'
    var_5 = {}
    var_6 = 'result.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja2 template.'
    var_1 = 'test_script.py'
    var_2 = '\ndata = {\n    \'name\': \'{{ project_name }}\',\n    \'version\': \'{{ version }}\',\n    \'items\': [{% for item in items %}\'{{ item }}\'{{ ", " if not loop.last else "" }}{% endfor %}]\n}\nwith open(\'{{ output_file }}\', \'w\') as f:\n    f.write(str(data))\n'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'version'
    var_6 = 'items'
    var_7 = 'output_file'
    var_8 = 'MyProject'
    var_9 = '1.0.0'
    var_10 = 'item1'
    var_11 = 'item2'
    var_12 = 'item3'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'output.txt'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    assert var_7 is None
    var_8 = 'pre_gen_project.py'
    var_9 = "#!/usr/bin/env python\nprint('test')"
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'pre_gen_project.sh'
    var_15 = "#!/bin/bash\necho 'test'"
    var_16 = 'pre_gen_project'
    var_17 = 'hooks'
    var_18 = module_0.find_hook(var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'pre_gen_project.py~'
    var_21 = "#!/usr/bin/env python\nprint('backup')"
    var_22 = 'pre_gen_project'
    var_23 = 'hooks'
    var_24 = module_0.find_hook(var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'unsupported_hook.py'
    var_27 = "#!/usr/bin/env python\nprint('unsupported')"
    var_28 = 'pre_gen_project'
    var_29 = 'hooks'
    var_30 = module_0.find_hook(var_28, var_29)
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 'post_gen_project.py'
    var_33 = "#!/usr/bin/env python\nprint('post')"
    var_34 = 'post_gen_project'
    var_35 = 'hooks'
    var_36 = module_0.find_hook(var_34, var_35)
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = 'non_existent_hook'
    var_39 = 'hooks'
    var_40 = module_0.find_hook(var_38, var_39)
    assert var_40 is None



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'post_gen_project'
    var_16 = False
    var_17 = 'Undefined variable'
    var_18 = 'post_gen_project'
    var_19 = True



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter.hooks.run_hook'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True
    var_13 = 'post_gen_project'
    var_14 = False
    var_15 = 'Undefined variable'
    var_16 = 'pre_gen_project'
    var_17 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\nexit 0\n'
    var_6 = 493
    var_7 = 'repo3'
    var_8 = '#!/bin/bash\nexit 1\n'
    var_9 = 'repo4'
    var_10 = 'pre_prompt.py'
    var_11 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)\n'
    var_12 = 'repo5'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when hook fails and delete_project_on_failure is False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when hook fails and delete_project_on_failure is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Hook failed'
    var_8 = 'cookiecutter.hooks.rmtree'
    var_9 = 'post_gen_project'
    var_10 = True

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when UndefinedError is raised.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'Undefined variable'
    var_8 = module_0.UndefinedError(var_7)
    var_9 = 'cookiecutter.hooks.rmtree'
    var_10 = 'pre_gen_project'
    var_11 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir changes working directory to repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes a script successfully.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(0)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit status.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes a bash script successfully.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on bash script failure.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 42'
    var_3 = 493
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles ENOEXEC error for empty or shebang-less files.'
    var_1 = 'test_script.sh'
    var_2 = ''
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles OSError exceptions.'
    var_1 = 'Popen'
    var_2 = 'test_script.py'
    var_3 = 'import sys\nsys.exit(0)'
    var_4 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes script from specified working directory.'
    var_1 = 'scripts'
    var_2 = 'work'
    var_3 = 'test_script.py'
    var_4 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'Test run_script calls make_executable on the script.'
    var_1 = 'cookiecutter.utils.make_executable'
    var_2 = 'test_script.py'
    var_3 = 'import sys\nsys.exit(0)'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = "# {{ cookiecutter.project_name }}\necho 'Hello {{ cookiecutter.author }}'"
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python file.'
    var_1 = "print('{{ cookiecutter.message }}')"
    var_2 = 'test_script.py'
    var_3 = 'cookiecutter'
    var_4 = 'message'
    var_5 = 'Success'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined Jinja variable.'
    var_1 = "echo '{{ undefined_var }}'"
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_script'

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja template.'
    var_1 = '#!/bin/bash\n# Project: {{ cookiecutter.project_name }}\n# Author: {{ cookiecutter.author }}\n{% if cookiecutter.use_docker %}\necho "Docker enabled"\n{% endif %}\n'
    var_2 = 'test_script.sh'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'use_docker'
    var_7 = 'my_project'
    var_8 = 'John Doe'
    var_9 = True
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test that run_script_with_context preserves file extension.'
    var_1 = "#!/usr/bin/env python\nprint('test')"
    var_2 = 'test_script.py'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = len(var_6)
    assert var_8 == 1



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_gen_project.py'
    var_10 = "#!/usr/bin/env python\nprint('Hook executed')"
    var_11 = 'pre_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when no hook exists.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'
    var_9 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_gen_project.py'
    var_10 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_11 = 'pre_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_gen_project.py'
    var_10 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_11 = 'pre_gen_project'
    var_12 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_gen_project.py'
    var_10 = "#!/usr/bin/env python\nprint('{{ undefined_var }}')"
    var_11 = 'pre_gen_project'
    var_12 = True



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with rendered context.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ project_name }}')\n"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = 'test_script.sh'
    var_2 = 'echo {{ var1 }} {{ var2 }}\n'
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'var1'
    var_5 = 'var2'
    var_6 = 'hello'
    var_7 = 'world'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 0
    var_10 = 'utf-8'

def test_case_0():
    var_0 = 'Test run_script_with_context preserves script extension.'
    var_1 = 'test_script.py'
    var_2 = '# {{ comment }}\n'
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'comment'
    var_5 = 'test comment'
    var_6 = {var_4: var_5}
    var_7 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined variables raises UndefinedError.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ undefined_var }}')\n"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'other_var'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.run_script_with_context(var_0, var_1, var_6)

def test_case_0():
    var_0 = 'Test run_script_with_context with empty context.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho 'test'\n"
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = {}

def test_case_0():
    var_0 = 'Test run_script_with_context accepts pathlib.Path objects.'
    var_1 = 'test_script.py'
    var_2 = '# {{ value }}\n'
    var_3 = 'cookiecutter.hooks.run_script'
    var_4 = 'value'
    var_5 = 'test'
    var_6 = {var_4: var_5}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project.py'
    var_10 = "#!/usr/bin/env python\nprint('success')"
    var_11 = 'post_gen_project'
    var_12 = False
    var_13 = '#!/usr/bin/env python\nexit(1)'
    var_14 = 'post_gen_project'
    var_15 = True
    var_16 = 'post_gen_project'
    var_17 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir when no hook exists.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'post_gen_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError and cleans up.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'post_gen_project.py'
    var_5 = "#!/usr/bin/env python\nprint('{{ undefined_var }}')"
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'post_gen_project'
    var_12 = True



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function executes scripts correctly.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Test hook\nprint('Hook executed')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'test_project'
    var_8 = 'test_author'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'hooks'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook renders context variables in hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Project: {{ project_name }}\nprint('{{ author }}')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'my_project'
    var_8 = 'john_doe'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook raises exception when script fails.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'import sys\nsys.exit(1)'
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Hook 1\nprint('First')"
    var_4 = 'pre_gen_project.sh'
    var_5 = "#!/bin/bash\necho 'Second'"
    var_6 = 'project'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = 'pre_gen_project'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'print("Hook executed")'
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when hook is not found.'
    var_1 = 'hooks'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when hook script fails.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'import sys\nsys.exit(1)'
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook with Jinja context rendering.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = '# Project: {{ cookiecutter.project_name }}\nprint("OK")'
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook with multiple hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'print("Hook 1")'
    var_4 = 'pre_gen_project.sh'
    var_5 = '#!/bin/bash\necho "Hook 2"'
    var_6 = 'project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'pre_gen_project'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = 493
    var_7 = 'repo3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'repo4'
    var_10 = 'pre_prompt.py'
    var_11 = 'import sys\nsys.exit(0)'
    var_12 = 'repo5'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts correctly.'
    var_1 = 'test_script.py'
    var_2 = "print('success')"

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit status.'
    var_1 = 'failing_script.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes shell scripts.'
    var_1 = 'test_script.bat'
    var_2 = '@echo off\nexit /b 0'
    var_3 = 'test_script.sh'
    var_4 = '#!/bin/bash\nexit 0'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test run_script raises FailedHookException when script doesn't exist."
    var_1 = '/nonexistent/path/script.py'
    var_2 = module_0.run_script(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException for empty file without shebang.'
    var_1 = 'empty_script.sh'
    var_2 = ''
    var_3 = module_0.run_script(var_1, var_2)

def test_case_0():
    var_0 = 'Test run_script executes in specified working directory.'
    var_1 = 'test_script.py'
    var_2 = "import os\nassert os.getcwd() == os.path.abspath('.')"

def test_case_0():
    var_0 = 'Test run_script uses sys.executable for .py files.'
    var_1 = 'test_script.py'
    var_2 = "print('python script')"



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts correctly.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(0)'

def test_case_0():
    var_0 = 'Test run_script executes non-python scripts.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = 'subprocess.Popen'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles OSError with ENOEXEC errno.'
    var_1 = 'test_script.py'
    var_2 = 'invalid'
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = 'subprocess.Popen'
    var_5 = 'Exec format error'
    var_6 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles other OSError exceptions.'
    var_1 = 'test_script.py'
    var_2 = 'test'
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = 'subprocess.Popen'
    var_5 = 'Permission denied'
    var_6 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script uses sys.executable for Python scripts.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = 'subprocess.Popen'

def test_case_0():
    var_0 = 'Test run_script uses shell on Windows.'
    var_1 = 'test_script.py'
    var_2 = 'test'
    var_3 = 'cookiecutter.utils.make_executable'
    var_4 = 'subprocess.Popen'
    var_5 = 'sys.platform'
    var_6 = 'win32'

def test_case_0():
    var_0 = 'Test run_script passes cwd parameter to subprocess.'
    var_1 = 'test_script.py'
    var_2 = 'test'
    var_3 = 'workdir'
    var_4 = 'cookiecutter.utils.make_executable'
    var_5 = 'subprocess.Popen'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test find_hook function with various scenarios.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt'
    var_4 = 'non_existent'
    var_5 = 'pre_prompt.py'
    var_6 = 'pre_prompt.sh~'
    var_7 = '~'
    var_8 = 'unsupported_hook.sh'
    var_9 = 'unsupported_hook'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook with default hooks directory.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt.sh'
    var_3 = 'pre_prompt'
    var_4 = module_0.find_hook(var_3, var_1)
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = 'Test find_hook with empty hooks directory.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script.'
    var_1 = 'print("{{ variable }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'variable'
    var_5 = 'hello_world'
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined variable raises UndefinedError.'
    var_1 = 'echo "{{ undefined_var }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.run_script_with_context(var_0, var_2, var_8)

def test_case_0():
    var_0 = 'Test run_script_with_context creates temporary file with correct extension.'
    var_1 = '#!/bin/bash\necho "{{ name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'test_name'
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context accepts Path objects.'
    var_1 = 'echo "{{ value }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'value'
    var_5 = 'test_value'
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = 493
    var_7 = 'repo3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'repo4'
    var_10 = 'pre_prompt.py'
    var_11 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'
    var_12 = 'repo5'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the valid_hook function with various inputs.'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt'
    var_3 = module_0.valid_hook(var_1, var_2)
    assert var_3 is True
    var_4 = 'pre_prompt.sh'
    var_5 = module_0.valid_hook(var_4, var_2)
    assert var_5 is True
    var_6 = 'pre_gen_project.py'
    var_7 = 'pre_gen_project'
    var_8 = module_0.valid_hook(var_6, var_7)
    assert var_8 is True
    var_9 = 'post_gen_project.sh'
    var_10 = 'post_gen_project'
    var_11 = module_0.valid_hook(var_9, var_10)
    assert var_11 is True
    var_12 = 'pre_prompt.py~'
    var_13 = module_0.valid_hook(var_12, var_2)
    assert var_13 is False
    var_14 = 'pre_gen_project.sh~'
    var_15 = module_0.valid_hook(var_14, var_7)
    assert var_15 is False
    var_16 = module_0.valid_hook(var_1, var_10)
    assert var_16 is False
    var_17 = 'pre_gen_project.sh'
    var_18 = module_0.valid_hook(var_17, var_2)
    assert var_18 is False
    var_19 = 'unsupported_hook.py'
    var_20 = 'unsupported_hook'
    var_21 = module_0.valid_hook(var_19, var_20)
    assert var_21 is False
    var_22 = 'invalid.sh'
    var_23 = 'invalid'
    var_24 = module_0.valid_hook(var_22, var_23)
    assert var_24 is False
    var_25 = 'pre_prompt.txt'
    var_26 = module_0.valid_hook(var_25, var_2)
    assert var_26 is True
    var_27 = module_0.valid_hook(var_2, var_2)
    assert var_27 is True
    var_28 = '/path/to/pre_prompt.py'
    var_29 = module_0.valid_hook(var_28, var_2)
    assert var_29 is True
    var_30 = 'subdir/pre_gen_project.sh'
    var_31 = module_0.valid_hook(var_30, var_7)
    assert var_31 is True
    var_32 = 'pre_prompt.py.bak'
    var_33 = module_0.valid_hook(var_32, var_2)
    assert var_33 is False
    var_34 = 'pre_prompt_old.py'
    var_35 = module_0.valid_hook(var_34, var_2)
    assert var_35 is False
    var_36 = 'pre_prompter.py'
    var_37 = module_0.valid_hook(var_36, var_2)
    assert var_37 is False
    var_38 = 'pre_prompt.py.py'
    var_39 = module_0.valid_hook(var_38, var_2)
    assert var_39 is True
    var_40 = 'pre_prompt.tar.gz'
    var_41 = module_0.valid_hook(var_40, var_2)
    assert var_41 is True



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'template_with_hook'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'template_fail'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'template_py'
    var_10 = 'pre_prompt.py'
    var_11 = "import sys\nprint('test')\nsys.exit(0)"
    var_12 = 'template_multi'
    var_13 = "#!/bin/bash\necho 'test1'"
    var_14 = "print('test2')"



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'template2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 'template3'
    var_7 = '#!/bin/bash\nexit 1'
    var_8 = 'template4'
    var_9 = 'pre_prompt.py'
    var_10 = "print('test')\n"
    var_11 = 'template5'
    var_12 = 'pre_prompt.sh~'



# Parsed testcases at query #24
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'pre_prompt.py'
    var_8 = 'print("test")'
    var_9 = len(var_3)
    assert var_9 == 2
    var_10 = 'pre_prompt.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_3)
    assert var_12 == 2
    var_13 = 'post_gen_project'
    var_14 = 'unsupported_hook.sh'
    var_15 = '#!/bin/bash\necho "unsupported"'
    var_16 = 'unsupported_hook'
    var_17 = 'readme.txt'
    var_18 = 'This is a readme'
    var_19 = len(var_3)
    assert var_19 == 2
    var_20 = 'post_gen_project.py'
    var_21 = 'print("post gen")'
    var_22 = len(var_3)
    assert var_22 == 1
    var_23 = 'pre_gen_project.sh'
    var_24 = '#!/bin/bash\necho "pre gen"'
    var_25 = 'pre_gen_project'
    var_26 = len(var_3)
    assert var_26 == 1



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/usr/bin/env python\n# Test script\nname = "{{ cookiecutter.project_name }}"\nversion = "{{ cookiecutter.version }}"\nprint(f"Project: {name}, Version: {version}")\n'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'version'
    var_8 = 'test_project'
    var_9 = '1.0.0'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 'MockPopen'
    var_13 = ()
    var_14 = 'wait'
    var_15 = 0
    var_16 = lambda self: var_15
    var_17 = {var_14: var_16}
    var_18 = type(var_12, var_13, var_17)
    var_19 = 'subprocess.Popen'

def test_case_0():
    var_0 = 'Test run_script_with_context raises UndefinedError for missing context.'
    var_1 = '#!/usr/bin/env python\n# Test script with undefined variable\nname = "{{ cookiecutter.missing_var }}"\n'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'Test run_script_with_context with shell script.'
    var_1 = '#!/bin/bash\necho "Project: {{ cookiecutter.project_name }}"\n'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'MockPopen'
    var_11 = ()
    var_12 = 'wait'
    var_13 = 0
    var_14 = lambda self: var_13
    var_15 = {var_12: var_14}
    var_16 = type(var_10, var_11, var_15)
    var_17 = 'subprocess.Popen'
    var_18 = 'cookiecutter.utils.make_executable'
    var_19 = None
    var_20 = lambda x: var_19

def test_case_0():
    var_0 = 'Test run_script_with_context propagates hook execution failures.'
    var_1 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)\n'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'MockPopen'
    var_11 = ()
    var_12 = 'wait'
    var_13 = 1
    var_14 = lambda self: var_13
    var_15 = {var_12: var_14}
    var_16 = type(var_10, var_11, var_15)
    var_17 = 'subprocess.Popen'
    var_18 = 'cookiecutter.utils.make_executable'
    var_19 = None
    var_20 = lambda x: var_19

def test_case_0():
    var_0 = 'Test run_script_with_context creates temporary file with correct extension.'
    var_1 = '{{ cookiecutter.name }}'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'rendered_content'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'MockPopen'
    var_11 = ()
    var_12 = 'wait'
    var_13 = 0
    var_14 = lambda self: var_13
    var_15 = {var_12: var_14}
    var_16 = type(var_10, var_11, var_15)
    var_17 = 'tempfile.NamedTemporaryFile'
    var_18 = 'subprocess.Popen'
    var_19 = 'cookiecutter.utils.make_executable'
    var_20 = None
    var_21 = lambda x: var_20
    var_22 = len(var_9)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 'repo3'
    var_7 = '#!/bin/bash\nexit 1'
    var_8 = 'repo4'
    var_9 = 'pre_prompt.py'
    var_10 = "#!/usr/bin/env python\nprint('test')"
    var_11 = 'repo5'
    var_12 = "#!/bin/bash\necho 'test1'"
    var_13 = "#!/usr/bin/env python\nprint('test2')"
    var_14 = 'repo6'
    var_15 = 'pre_prompt.sh~'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts correctly.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493

def test_case_0():
    var_0 = 'Test run_script executes Python scripts correctly.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(0)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = 493
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles ENOEXEC error.'
    var_1 = 'test_script.sh'
    var_2 = 'invalid script content'
    var_3 = 'subprocess.Popen'
    var_4 = 'Exec format error'
    var_5 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles other OSError.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'subprocess.Popen'
    var_4 = 'Permission denied'
    var_5 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script runs from specified working directory.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493
    var_4 = 'subprocess.Popen'
    var_5 = 'work_dir'
    var_6 = 1

def test_case_0():
    var_0 = 'Test run_script uses shell on Windows.'
    var_1 = 'test_script.bat'
    var_2 = '@echo off\nexit /b 0'
    var_3 = 'sys.platform'
    var_4 = 'win32'
    var_5 = 'subprocess.Popen'
    var_6 = 1

def test_case_0():
    var_0 = "Test run_script doesn't use shell on Unix."
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 493
    var_4 = 'sys.platform'
    var_5 = 'linux'
    var_6 = 'subprocess.Popen'
    var_7 = 1



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = 'test_script.py'
    var_2 = '#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nname = "{{ cookiecutter.project_name }}"\nversion = "{{ cookiecutter.version }}"\nwith open("output.txt", "w") as f:\n    f.write(f"{name}-{version}")\n'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'test_project'
    var_8 = '1.0.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'output.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with shell script.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}" > output.txt\n'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'shell_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'output.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = 'test_script.py'
    var_2 = '#!/usr/bin/env python\nresult = "{{ var1 }}_{{ var2 }}_{{ var3 }}"\nwith open("result.txt", "w") as f:\n    f.write(result)\n'
    var_3 = 'utf-8'
    var_4 = 'var1'
    var_5 = 'var2'
    var_6 = 'var3'
    var_7 = 'alpha'
    var_8 = 'beta'
    var_9 = 'gamma'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = 'result.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with nested context variables.'
    var_1 = 'test_script.py'
    var_2 = '#!/usr/bin/env python\ndata = "{{ config.name }}_{{ config.settings.debug }}"\nwith open("nested.txt", "w") as f:\n    f.write(data)\n'
    var_3 = 'utf-8'
    var_4 = 'config'
    var_5 = 'name'
    var_6 = 'settings'
    var_7 = 'myapp'
    var_8 = 'debug'
    var_9 = 'true'
    var_10 = {var_8: var_9}
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'nested.txt'

def test_case_0():
    var_0 = 'Test that run_script_with_context preserves file extension.'
    var_1 = 'test_script.custom'
    var_2 = '#!/usr/bin/env python\nwith open("custom.txt", "w") as f:\n    f.write("{{ extension_test }}")\n'
    var_3 = 'utf-8'
    var_4 = 'extension_test'
    var_5 = 'success'
    var_6 = {var_4: var_5}
    var_7 = 'custom.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with empty context.'
    var_1 = 'test_script.py'
    var_2 = '#!/usr/bin/env python\nwith open("empty.txt", "w") as f:\n    f.write("no_context")\n'
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = 'empty.txt'

def test_case_0():
    var_0 = 'Test run_script_with_context with special characters in context.'
    var_1 = 'test_script.py'
    var_2 = '#!/usr/bin/env python\nwith open("special.txt", "w") as f:\n    f.write("{{ special_text }}")\n'
    var_3 = 'utf-8'
    var_4 = 'special_text'
    var_5 = 'hello-world_123!@#'
    var_6 = {var_4: var_5}
    var_7 = 'special.txt'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts correctly.'
    var_1 = 'test_script.py'
    var_2 = "print('success')"

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(1)'
    var_3 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes shell scripts.'
    var_1 = 'test_script.sh'
    var_2 = '@echo off\nexit /b 0'
    var_3 = '#!/bin/bash\nexit 0'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles ENOEXEC error.'
    var_1 = 'test_script.sh'
    var_2 = ''
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles OSError.'
    var_1 = 'nonexistent.py'
    var_2 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script respects cwd parameter.'
    var_1 = 'test_script.py'
    var_2 = "import os\nassert os.getcwd() == os.path.abspath('.')"

def test_case_0():
    var_0 = 'Test run_script makes script executable.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'make_executable'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.py'
    var_5 = "#!/usr/bin/env python\nprint('hook executed')"
    var_6 = 'repo3'
    var_7 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_8 = 'repo4'
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'bash hook executed'"



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'repo3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'repo4'
    var_10 = 'pre_prompt.py'
    var_11 = "#!/usr/bin/env python\nprint('test')"
    var_12 = 'repo5'
    var_13 = "#!/bin/bash\necho 'script1'"
    var_14 = "#!/usr/bin/env python\nprint('script2')"



# Parsed testcases at query #32
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'pre_prompt.py'
    var_8 = 'print("test")'
    var_9 = len(var_3)
    assert var_9 == 2
    var_10 = 'pre_prompt.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_3)
    assert var_12 == 2
    var_13 = 'unsupported_hook.sh'
    var_14 = '#!/bin/bash\necho "unsupported"'
    var_15 = 'unsupported_hook'
    var_16 = 'post_gen_project.py'
    var_17 = 'print("post gen")'
    var_18 = len(var_3)
    assert var_18 == 2
    var_19 = 'post_gen_project'
    var_20 = len(var_3)
    assert var_20 == 1



# Parsed testcases at query #33
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = '/path/to/repo'
    var_2 = '/path/to/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_8)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on FailedHookException.'
    var_1 = '/path/to/repo'
    var_2 = '/path/to/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = True
    var_11 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_10)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up project on UndefinedError.'
    var_1 = '/path/to/repo'
    var_2 = '/path/to/project'
    var_3 = 'pre_gen_project'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = 'Variable undefined'
    var_10 = True
    var_11 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_10)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't clean up when delete_project_on_failure is False."
    var_1 = '/path/to/repo'
    var_2 = '/path/to/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = False
    var_11 = module_0.run_hook_from_repo_dir(var_1, var_3, var_2, var_6, var_10)



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with Jinja rendering.'
    var_1 = '#!/bin/bash\necho {{ cookiecutter.project_name }}'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with Python script.'
    var_1 = "print('{{ cookiecutter.name }}')"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'my_app'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0
    var_11 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined template variable.'
    var_1 = '#!/bin/bash\necho {{ undefined_var }}'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja template.'
    var_1 = '#!/bin/bash\n{% if cookiecutter.use_feature %}echo Feature enabled{% endif %}\necho {{ cookiecutter.version }}'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'use_feature'
    var_7 = 'version'
    var_8 = True
    var_9 = '1.0.0'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 0

def test_case_0():
    var_0 = 'Test that run_script_with_context preserves file extension.'
    var_1 = 'test_script.bash'
    var_2 = '#!/bin/bash\necho {{ var }}'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'var'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = '.bash'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function executes scripts correctly.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "print('test hook')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'cookiecutter'
    var_7 = 'test_project'
    var_8 = {var_5: var_7}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'pre_gen_project'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook returns early when no scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = 'cookiecutter.hooks.run_script_with_context'
    var_3 = 'pre_gen_project'
    var_4 = '/some/path'
    var_5 = {}
    var_6 = module_0.run_hook(var_3, var_4, var_5)

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'pre_gen_project.sh'
    var_4 = "print('hook1')"
    var_5 = "#!/bin/bash\necho 'hook2'"
    var_6 = 'project'
    var_7 = 'test'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'pre_gen_project'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'repo3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'repo4'
    var_10 = '#!/bin/bash\nexit 0'
    var_11 = 'pre_prompt.py'
    var_12 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'



# Parsed testcases at query #37
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("test")'
    var_9 = len(var_2)
    assert var_9 == 2
    var_10 = 'pre_gen_project.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_2)
    assert var_12 == 2
    var_13 = '~'
    var_14 = 'non_existent_hook'
    var_15 = 'unsupported_hook.sh'
    var_16 = '#!/bin/bash'
    var_17 = 'unsupported_hook'
    var_18 = 'pre_prompt.py'
    var_19 = 'print("prompt")'
    var_20 = 'post_gen_project.sh'
    var_21 = 'pre_prompt'
    var_22 = 'post_gen_project'
    var_23 = 'nonexistent'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts successfully.'
    var_1 = 'test_script.py'
    var_2 = "print('Hello')"

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit.'
    var_1 = 'import sys\nsys.exit(1)'

def test_case_0():
    var_0 = 'Test run_script executes shell scripts on non-Windows.'
    var_1 = 'Shell script test not applicable on Windows'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/bash\nexit 0'
    var_4 = 493

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on ENOEXEC error.'
    var_1 = 'test_script.sh'
    var_2 = ''
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on OSError.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes with specified working directory.'
    var_1 = 'test_script.py'
    var_2 = "import os\nassert os.getcwd() == r'"
    var_3 = "'"

def test_case_0():
    var_0 = 'Test run_script calls make_executable.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = []
    var_4 = 'make_executable'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on FailedHookException.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = 'Hook failed'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir cleans up on UndefinedError.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = 'Undefined variable'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir does not clean up when flag is False.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = 'Hook failed'
    var_9 = False



# Parsed testcases at query #40
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project'
    var_3 = 'hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None
    var_5 = 'pre_gen_project'
    var_6 = 'hooks'
    var_7 = module_0.find_hook(var_5, var_6)
    assert var_7 is None
    var_8 = 'pre_gen_project.sh'
    var_9 = "#!/bin/bash\necho 'test'"
    var_10 = 'pre_gen_project'
    var_11 = 'hooks'
    var_12 = module_0.find_hook(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = str(var_9)
    var_15 = 'pre_gen_project.py'
    var_16 = "print('test')"
    var_17 = 'pre_gen_project'
    var_18 = 'hooks'
    var_19 = module_0.find_hook(var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'pre_gen_project.sh~'
    var_22 = "#!/bin/bash\necho 'backup'"
    var_23 = 'pre_gen_project'
    var_24 = 'hooks'
    var_25 = module_0.find_hook(var_23, var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = str(var_9)
    var_28 = 'post_gen_project.sh'
    var_29 = "#!/bin/bash\necho 'other'"
    var_30 = 'pre_gen_project'
    var_31 = 'hooks'
    var_32 = module_0.find_hook(var_30, var_31)
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = str(var_9)
    var_35 = 'unsupported_hook'
    var_36 = 'hooks'
    var_37 = module_0.find_hook(var_35, var_36)
    assert var_37 is None
    var_38 = 'custom_hooks'
    var_39 = 'pre_prompt.sh'
    var_40 = "#!/bin/bash\necho 'custom'"
    var_41 = 'pre_prompt'
    var_42 = 'custom_hooks'
    var_43 = module_0.find_hook(var_41, var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = str(var_9)



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts successfully.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'

def test_case_0():
    var_0 = 'Test run_script executes Python scripts.'
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(0)'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on ENOEXEC error.'
    var_1 = 'test_script'
    var_2 = ''
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on OSError.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = "Test run_script uses default cwd of '.'."
    var_1 = 'test_script.py'
    var_2 = 'import sys\nsys.exit(0)'
    var_3 = module_0.run_script(var_1)

def test_case_0():
    var_0 = 'Test run_script uses shell on Windows.'
    var_1 = 'platform'
    var_2 = 'win32'
    var_3 = 'test_script.bat'
    var_4 = '@echo off\nexit /b 0'
    var_5 = []
    var_6 = 'Popen'
    var_7 = len(var_5)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir function.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'post_gen_project.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = 493
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with no hooks present.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'post_gen_project.sh'
    var_5 = '#!/bin/bash\nexit 1'
    var_6 = 493
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'post_gen_project.sh'
    var_5 = '#!/bin/bash\nexit 1'
    var_6 = 493
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = False

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir with UndefinedError in template rendering.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'hooks'
    var_4 = 'post_gen_project.sh'
    var_5 = '#!/bin/bash\necho {{ undefined_var }}'
    var_6 = 493
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'post_gen_project'
    var_13 = True



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.hooks.run_hook'
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = 'Hook failed'
    var_10 = 'cookiecutter.hooks.rmtree'
    var_11 = 'post_gen_project'
    var_12 = True
    var_13 = 'Variable undefined'
    var_14 = 'pre_gen_project'
    var_15 = False
    var_16 = 'pre_gen_project'
    var_17 = True



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'post_gen_project'
    var_16 = False
    var_17 = 'Variable undefined'
    var_18 = 'post_gen_project'
    var_19 = True



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook executes scripts found in hooks directory.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.sh'
    var_3 = '#!/bin/bash\necho "test"'
    var_4 = 493
    var_5 = 'project'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.hooks.find_hook'
    var_12 = 'pre_gen_project'
    var_13 = None
    var_14 = []
    var_15 = 'cookiecutter.hooks.run_script_with_context'
    var_16 = len(var_14)
    assert var_16 == 1

def test_case_0():
    var_0 = 'Test run_hook returns early when no scripts are found.'
    var_1 = 'project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = None
    var_9 = lambda hook_name: var_8
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'pre_gen_project'
    var_13 = len(var_10)
    assert var_13 == 0

def test_case_0():
    var_0 = 'Test run_hook executes multiple scripts in order.'
    var_1 = 'hooks'
    var_2 = 'post_gen_project.sh'
    var_3 = '#!/bin/bash\necho "test1"'
    var_4 = 'post_gen_project.py'
    var_5 = 'print("test2")'
    var_6 = 'project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.hooks.find_hook'
    var_13 = 'post_gen_project'
    var_14 = None
    var_15 = []
    var_16 = 'cookiecutter.hooks.run_script_with_context'
    var_17 = len(var_15)
    assert var_17 == 2



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'repo'
    var_2 = 'hooks'
    var_3 = 'repo2'
    var_4 = 'pre_prompt.py'
    var_5 = "#!/usr/bin/env python\nprint('hook executed')"
    var_6 = 'repo3'
    var_7 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)'
    var_8 = 'repo4'
    var_9 = 'pre_prompt.sh'
    var_10 = "#!/bin/bash\necho 'hook executed'"
    var_11 = 'repo5'
    var_12 = "#!/usr/bin/env python\nprint('first hook')"
    var_13 = "#!/bin/bash\necho 'second hook'"



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test find_hook function with various scenarios.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = 'nonexistent'
    var_4 = 'pre_prompt.py'
    var_5 = 0
    var_6 = 'pre_prompt.sh'
    var_7 = 'pre_prompt.py~'
    var_8 = 'post_gen_project'
    var_9 = 'unsupported_hook.py'
    var_10 = 'unsupported_hook'
    var_11 = 'post_gen_project.py'
    var_12 = 'pre_gen_project.sh'
    var_13 = 'pre_gen_project'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script.'
    var_1 = 'print("{{ project_name }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = '#!/usr/bin/env python3\nprint("{{ value }}")'
    var_2 = 'hook.py'
    var_3 = 'utf-8'
    var_4 = 'value'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'cookiecutter.hooks.run_script'
    var_8 = 0
    var_9 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja2 template.'
    var_1 = '#!/bin/bash\n{% for item in items %}\necho "{{ item }}"\n{% endfor %}\necho "{{ config.name }}"\n'
    var_2 = 'complex_script.sh'
    var_3 = 'utf-8'
    var_4 = 'items'
    var_5 = 'config'
    var_6 = 'item1'
    var_7 = 'item2'
    var_8 = 'item3'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'name'
    var_11 = 'myconfig'
    var_12 = {var_10: var_11}
    var_13 = {var_4: var_9, var_5: var_12}
    var_14 = 'cookiecutter.hooks.run_script'
    var_15 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context passes cwd to run_script.'
    var_1 = 'echo "test"'
    var_2 = 'test.sh'
    var_3 = 'utf-8'
    var_4 = 'subdir'
    var_5 = {}
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = 0



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'post_gen_project'
    var_14 = False
    var_15 = 'post_gen_project'
    var_16 = True
    var_17 = 'Undefined variable'
    var_18 = 'post_gen_project'
    var_19 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'template2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'template3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'template4'
    var_10 = 'pre_prompt.py'
    var_11 = "#!/usr/bin/env python\nprint('test')"



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes a script successfully.'
    var_1 = 'test_script.py'
    var_2 = "print('Hello World')\n"

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on non-zero exit status.'
    var_1 = 'failing_script.py'
    var_2 = 'import sys\nsys.exit(1)\n'
    var_3 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes shell script on non-Windows.'
    var_1 = 'Shell script test skipped on Windows'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/bash\nexit 0\n'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on ENOEXEC error.'
    var_1 = 'test_script.py'
    var_2 = 'invalid script'
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises FailedHookException on OSError.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes script from specified working directory.'
    var_1 = 'scripts'
    var_2 = 'test_script.py'
    var_3 = 'import os\nassert os.getcwd() == os.path.dirname(os.path.abspath(__file__))\n'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with Jinja rendering.'
    var_1 = "#!/usr/bin/env python\nprint('{{ project_name }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context with bash script.'
    var_1 = "#!/bin/bash\necho '{{ project_name }}'\n"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined Jinja variable.'
    var_1 = "#!/usr/bin/env python\nprint('{{ undefined_var }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = "#!/usr/bin/env python\nprint('{{ name }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja template.'
    var_1 = "#!/usr/bin/env python\n{% for item in items %}\nprint('{{ item }}')\n{% endfor %}\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'items'
    var_5 = 'item1'
    var_6 = 'item2'
    var_7 = 'item3'
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_4: var_8}

def test_case_0():
    var_0 = 'Test run_script_with_context raises FailedHookException on script failure.'
    var_1 = '#!/usr/bin/env python\nimport sys\nsys.exit(1)\n'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = {}

def test_case_0():
    var_0 = 'Test run_script_with_context works with string path.'
    var_1 = "#!/usr/bin/env python\nprint('test')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = {}

def test_case_0():
    var_0 = 'Test run_script_with_context with empty context.'
    var_1 = "#!/usr/bin/env python\nprint('hello')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = {}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles FailedHookException and deletes project.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = 'Hook failed'
    var_9 = True

def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir handles UndefinedError and deletes project.'
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = 'Undefined variable'
    var_9 = True

def test_case_0():
    var_0 = "Test run_hook_from_repo_dir doesn't delete project when delete_project_on_failure is False."
    var_1 = '/repo'
    var_2 = '/project'
    var_3 = 'post_gen_project'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = 'Hook failed'
    var_9 = False



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function executes hook scripts with context.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "import os\nwith open('{{ cookiecutter.output_file }}', 'w') as f:\n    f.write('{{ cookiecutter.project_name }}')\n"
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'output_file'
    var_8 = 'test_project'
    var_9 = 'test_output.txt'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'hooks'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook raises exception when hook script fails.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "raise RuntimeError('Hook failed')\n"
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook with bash script.'
    var_1 = 'Bash scripts not supported on Windows'
    var_2 = 'hooks'
    var_3 = 'post_gen_project.sh'
    var_4 = "#!/bin/bash\necho 'hook executed' > output.txt\n"
    var_5 = 493
    var_6 = 'project'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'post_gen_project'
    var_11 = 'output.txt'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts in order.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "with open('execution_order.txt', 'a') as f:\n    f.write('script1\\n')\n"
    var_4 = 'pre_gen_project_extra.py'
    var_5 = "# This won't match hook name\npass\n"
    var_6 = 'project'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'pre_gen_project'
    var_11 = 'execution_order.txt'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function executes scripts found by find_hook.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Test hook\nprint('Hook executed')\n"
    var_4 = 493
    var_5 = 'project'
    var_6 = 'cookiecutter.hooks.find_hook'
    var_7 = 'pre_gen_project'
    var_8 = None
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script_with_context'
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = 'test_project'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = len(var_9)
    assert var_16 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'cookiecutter.hooks.find_hook'
    var_2 = None
    var_3 = lambda hook_name: var_2
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'pre_gen_project'
    var_10 = '/tmp/project'
    var_11 = module_0.run_hook(var_9, var_10, var_8)

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = '# Hook 1\n'
    var_4 = 493
    var_5 = 'pre_gen_project.sh'
    var_6 = '#!/bin/bash\n# Hook 2\n'
    var_7 = 'project'
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = 'pre_gen_project'
    var_10 = None
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script_with_context'
    var_13 = 'cookiecutter'
    var_14 = 'project_name'
    var_15 = 'test_project'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = len(var_11)
    assert var_18 == 2



# Parsed testcases at query #11
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'pre_prompt.py'
    var_8 = 'print("test")'
    var_9 = len(var_3)
    assert var_9 == 2
    var_10 = 'pre_prompt.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_3)
    assert var_12 == 2
    var_13 = 'invalid_hook'
    var_14 = 'post_gen_project.sh'
    var_15 = '#!/bin/bash\necho "post"'
    var_16 = len(var_3)
    assert var_16 == 2
    var_17 = 'post_gen_project'
    var_18 = len(var_3)
    assert var_18 == 1
    var_19 = 'nonexistent_hooks'
    var_20 = module_0.find_hook(var_1, var_19)
    assert var_20 is None



# Parsed testcases at query #12
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test the valid_hook function.'
    var_1 = 'pre_prompt.py'
    var_2 = 'pre_prompt'
    var_3 = module_0.valid_hook(var_1, var_2)
    assert var_3 is True
    var_4 = 'pre_gen_project.sh'
    var_5 = 'pre_gen_project'
    var_6 = module_0.valid_hook(var_4, var_5)
    assert var_6 is True
    var_7 = 'post_gen_project.bat'
    var_8 = 'post_gen_project'
    var_9 = module_0.valid_hook(var_7, var_8)
    assert var_9 is True
    var_10 = module_0.valid_hook(var_2, var_2)
    assert var_10 is True
    var_11 = module_0.valid_hook(var_5, var_5)
    assert var_11 is True
    var_12 = module_0.valid_hook(var_1, var_8)
    assert var_12 is False
    var_13 = module_0.valid_hook(var_4, var_2)
    assert var_13 is False
    var_14 = 'unsupported_hook.py'
    var_15 = 'unsupported_hook'
    var_16 = module_0.valid_hook(var_14, var_15)
    assert var_16 is False
    var_17 = 'invalid_hook.sh'
    var_18 = 'invalid_hook'
    var_19 = module_0.valid_hook(var_17, var_18)
    assert var_19 is False
    var_20 = 'pre_prompt.py~'
    var_21 = module_0.valid_hook(var_20, var_2)
    assert var_21 is False
    var_22 = 'pre_gen_project.sh~'
    var_23 = module_0.valid_hook(var_22, var_5)
    assert var_23 is False
    var_24 = 'post_gen_project~'
    var_25 = module_0.valid_hook(var_24, var_8)
    assert var_25 is False
    var_26 = '/path/to/pre_prompt.py'
    var_27 = module_0.valid_hook(var_26, var_2)
    assert var_27 is True
    var_28 = 'hooks/pre_gen_project.sh'
    var_29 = module_0.valid_hook(var_28, var_5)
    assert var_29 is True
    var_30 = 'Pre_Prompt.py'
    var_31 = module_0.valid_hook(var_30, var_2)
    assert var_31 is False
    var_32 = 'PRE_PROMPT.py'
    var_33 = module_0.valid_hook(var_32, var_2)
    assert var_33 is False
    var_34 = '.pre_prompt'
    var_35 = module_0.valid_hook(var_34, var_2)
    assert var_35 is False
    var_36 = 'pre_prompt.'
    var_37 = module_0.valid_hook(var_36, var_2)
    assert var_37 is True
    var_38 = 'pre_prompt.backup.py'
    var_39 = module_0.valid_hook(var_38, var_2)
    assert var_39 is False
    var_40 = 'pre_prompt.test.sh'
    var_41 = module_0.valid_hook(var_40, var_2)
    assert var_41 is False



# Parsed testcases at query #13
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function with various scenarios.'
    var_1 = 'hooks'
    var_2 = 'pre_prompt'
    var_3 = 'nonexistent_hooks'
    var_4 = module_0.find_hook(var_2, var_3)
    assert var_4 is None
    var_5 = 'pre_prompt'
    var_6 = module_0.find_hook(var_5, var_3)
    assert var_6 is None
    var_7 = 'pre_prompt.sh'
    var_8 = "#!/bin/bash\necho 'test'"
    var_9 = 'pre_prompt'
    var_10 = module_0.find_hook(var_9, var_3)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'pre_prompt.py'
    var_13 = "print('test')"
    var_14 = 'pre_prompt'
    var_15 = module_0.find_hook(var_14, var_3)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 'pre_prompt.sh~'
    var_18 = "#!/bin/bash\necho 'backup'"
    var_19 = 'pre_prompt'
    var_20 = module_0.find_hook(var_19, var_3)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'post_gen_project'
    var_23 = module_0.find_hook(var_22, var_3)
    assert var_23 is None
    var_24 = 'unsupported_hook.sh'
    var_25 = 'unsupported_hook'
    var_26 = module_0.find_hook(var_25, var_3)
    assert var_26 is None
    var_27 = 'post_gen_project'
    var_28 = 'post_gen_project'
    var_29 = module_0.find_hook(var_28, var_3)
    var_30 = len(var_29)
    assert var_30 == 1



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hooks and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'pre_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'cookiecutter.hooks.logger'
    var_14 = 'pre_gen_project'
    var_15 = True
    var_16 = 'Undefined variable'
    var_17 = 'post_gen_project'
    var_18 = False



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and handles failures.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'post_gen_project'
    var_16 = False
    var_17 = 'Variable undefined'
    var_18 = 'post_gen_project'
    var_19 = True



# Parsed testcases at query #16
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script function executes scripts correctly.'
    var_1 = 'test_script.py'
    var_2 = 'exit(0)'
    var_3 = 'test_script_fail.py'
    var_4 = 'exit(1)'
    var_5 = module_0.run_script(var_0, var_1)
    var_6 = 'test_script.bat'
    var_7 = '@echo off\nexit /b 0'
    var_8 = 'test_script.sh'
    var_9 = '#!/bin/bash\nexit 0'
    var_10 = 'empty_script.sh'
    var_11 = ''
    var_12 = module_0.run_script(var_10, var_11)
    var_13 = 'test_script_default.py'
    var_14 = 'nonexistent.py'
    var_15 = module_0.run_script(var_10, var_11)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'test_repo2'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'cookiecutter.hooks.create_tmp_repo_dir'
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = 'test_repo3'
    var_10 = '#!/bin/bash\nexit 1'
    var_11 = 'Hook failed'
    var_12 = 'test_repo4'
    var_13 = 'pre_prompt.py'
    var_14 = "print('test')"
    var_15 = 'test_repo5'
    var_16 = "#!/bin/bash\necho 'test1'"
    var_17 = "print('test2')"



# Parsed testcases at query #18
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook function executes hook scripts found in hooks directory.'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '/path/to/hooks/pre_gen_project.sh'
    var_9 = module_0.run_hook(var_1, var_2, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'post_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.run_hook(var_1, var_2, var_7)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '/path/to/hooks/pre_gen_project.sh'
    var_9 = '/path/to/hooks/pre_gen_project.py'
    var_10 = [var_8, var_9]
    var_11 = module_0.run_hook(var_1, var_2, var_7)
    var_12 = 0
    var_13 = var_10[var_12]
    var_14 = 1
    var_15 = var_10[var_14]

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when find_hook returns empty list.'
    var_1 = 'pre_prompt'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.run_hook(var_1, var_2, var_7)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script file.'
    var_1 = 'print("{{ variable }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'variable'
    var_6 = 'test_value'
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = '.py'

def test_case_0():
    var_0 = 'Test run_script_with_context handles undefined variables gracefully.'
    var_1 = '#!/bin/bash\necho "{{ undefined_var }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'other_var'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct cwd to run_script.'
    var_1 = 'echo "test"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'custom_dir'
    var_6 = {}

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja2 template.'
    var_1 = '#!/bin/bash\n{% for item in items %}\necho "{{ item }}"\n{% endfor %}\necho "{{ name }}"\n'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'items'
    var_6 = 'name'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'test'
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = 0



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test run_script executes scripts correctly.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script executes Python scripts correctly.'
    var_1 = 'import sys\nsys.exit(0)'
    var_2 = module_0.run_script(var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises exception on non-zero exit status.'
    var_1 = 'failing_script.sh'
    var_2 = '#!/bin/bash\nexit 1'
    var_3 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script raises exception for script with missing shebang.'
    var_1 = 'no_shebang.sh'
    var_2 = 'echo hello'
    var_3 = module_0.run_script(var_0, var_1)

def test_case_0():
    var_0 = 'Test run_script executes script in specified working directory.'
    var_1 = 'work'
    var_2 = 'test_script.sh'
    var_3 = '#!/bin/bash\ntouch output.txt\nexit 0'
    var_4 = 'output.txt'

def test_case_0():
    var_0 = 'Test run_script executes Python scripts with proper exit handling.'
    var_1 = 'test_script.py'
    var_2 = "print('Hello')\nexit(0)"

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles OSError exceptions.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script handles generic OSError exceptions.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\nexit 0'
    var_3 = 'Popen'
    var_4 = module_0.run_script(var_0, var_1)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = []
    var_11 = None
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_10)
    assert var_13 == 1
    var_14 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script.'
    var_1 = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'myproject'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja2 template.'
    var_1 = '#!/bin/bash\n{% if cookiecutter.use_docker %}\ndocker build -t {{ cookiecutter.project_name }} .\n{% endif %}\necho "Project: {{ cookiecutter.project_name }}"\n'
    var_2 = 'complex_script.sh'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'use_docker'
    var_8 = 'my_app'
    var_9 = True
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context accepts Path objects.'
    var_1 = '#!/bin/bash\necho "{{ cookiecutter.value }}"'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = 'value'
    var_7 = 'test_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = '#!/usr/bin/env python\nprint("test")'
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function executes hook scripts correctly.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Test hook\nprint('Hook executed')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'test_project'
    var_8 = 'Test Author'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook when no hook scripts are found.'
    var_1 = 'hooks'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook renders context variables in hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "# Project: {{ project_name }}\nprint('{{ project_name }}')"
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'my_project'
    var_8 = 'Test'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook raises FailedHookException on script failure.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = 'import sys\nsys.exit(1)'
    var_4 = 'project'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts in sequence.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "print('Hook 1')"
    var_4 = 'pre_gen_project.sh'
    var_5 = "#!/bin/bash\necho 'Hook 2'"
    var_6 = 'project'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = 'pre_gen_project'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'test_repo'
    var_2 = 'hooks'
    var_3 = 'test_repo_with_hook'
    var_4 = 'pre_prompt.sh'
    var_5 = "#!/bin/bash\necho 'test'"
    var_6 = 493
    var_7 = 'test_repo_failing_hook'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'test_repo_python_hook'
    var_10 = 'pre_prompt.py'
    var_11 = "print('test')"
    var_12 = 'test_repo_multiple_hooks'
    var_13 = "#!/bin/bash\necho 'test1'"
    var_14 = "print('test2')"



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook_from_repo_dir executes hook and cleans up on failure.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.hooks.run_hook'
    var_9 = 'post_gen_project'
    var_10 = False
    var_11 = 'Hook failed'
    var_12 = 'cookiecutter.hooks.rmtree'
    var_13 = 'post_gen_project'
    var_14 = True
    var_15 = 'post_gen_project'
    var_16 = False
    var_17 = 'Variable undefined'
    var_18 = 'post_gen_project'
    var_19 = True



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with Jinja rendering.'
    var_1 = "#!/usr/bin/env python\nprint('{{ project_name }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context raises UndefinedError for missing context.'
    var_1 = "#!/usr/bin/env python\nprint('{{ missing_var }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context with bash script extension.'
    var_1 = "#!/bin/bash\necho '{{ project_name }}'\n"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'bash_project'
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_7[var_10][var_10]
    var_12 = '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = "#!/usr/bin/env python\nprint('{{ name }}-{{ version }}')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'name'
    var_5 = 'version'
    var_6 = 'myapp'
    var_7 = '1.0.0'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = []
    var_10 = 'cookiecutter.hooks.run_script'
    var_11 = len(var_9)
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with empty context.'
    var_1 = "#!/usr/bin/env python\nprint('hello')\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = []
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja logic.'
    var_1 = "#!/usr/bin/env python\n{% if debug %}\nprint('Debug mode')\n{% else %}\nprint('Production mode')\n{% endif %}\n"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'debug'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = []
    var_8 = 'cookiecutter.hooks.run_script'
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test run_pre_prompt_hook function.'
    var_1 = 'template'
    var_2 = 'hooks'
    var_3 = 'template2'
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\nexit 0'
    var_6 = 493
    var_7 = 'template3'
    var_8 = '#!/bin/bash\nexit 1'
    var_9 = 'template4'
    var_10 = 'pre_prompt.py'
    var_11 = '#!/usr/bin/env python\nimport sys\nsys.exit(0)'
    var_12 = 'template5'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook executes hook scripts found in the hooks directory.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "print('Hook executed')"
    var_4 = 'project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = 'cookiecutter.hooks.run_script_with_context'
    var_12 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook returns early when no hook scripts are found.'
    var_1 = 'project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter.hooks.find_hook'
    var_8 = None
    var_9 = 'cookiecutter.hooks.run_script_with_context'
    var_10 = 'pre_gen_project'

def test_case_0():
    var_0 = 'Test run_hook executes multiple hook scripts.'
    var_1 = 'hooks'
    var_2 = 'pre_gen_project.py'
    var_3 = "print('Hook 1')"
    var_4 = 'pre_gen_project.sh'
    var_5 = "#!/bin/bash\necho 'Hook 2'"
    var_6 = 'project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.hooks.find_hook'
    var_13 = 'cookiecutter.hooks.run_script_with_context'
    var_14 = 'pre_gen_project'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with rendered context.'
    var_1 = 'test_script.py'
    var_2 = "\nimport os\nwith open('{{ output_file }}', 'w') as f:\n    f.write('{{ greeting }} {{ name }}')\n"
    var_3 = 'utf-8'
    var_4 = 'output_file'
    var_5 = 'greeting'
    var_6 = 'name'
    var_7 = 'output.txt'
    var_8 = 'Hello'
    var_9 = 'World'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}

def test_case_0():
    var_0 = 'Test run_script_with_context with a shell script.'
    var_1 = 'test_script.sh'
    var_2 = '#!/bin/bash\necho "{{ message }}" > {{ output_file }}\n'
    var_3 = 'utf-8'
    var_4 = 'message'
    var_5 = 'output_file'
    var_6 = 'Test message'
    var_7 = 'shell_output.txt'
    var_8 = {var_4: var_6, var_5: var_7}

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja template.'
    var_1 = 'test_script.py'
    var_2 = "\nresult = '{% if condition %}yes{% else %}no{% endif %}'\nwith open('{{ filename }}', 'w') as f:\n    f.write(result)\n"
    var_3 = 'utf-8'
    var_4 = 'condition'
    var_5 = 'filename'
    var_6 = True
    var_7 = 'result.txt'
    var_8 = {var_4: var_6, var_5: var_7}

def test_case_0():
    var_0 = 'Test that run_script_with_context preserves file extension.'
    var_1 = 'test_script.py'
    var_2 = "\nwith open('{{ output }}', 'w') as f:\n    f.write('done')\n"
    var_3 = 'utf-8'
    var_4 = 'output'
    var_5 = 'done.txt'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test run_script_with_context with special characters in context.'
    var_1 = 'test_script.py'
    var_2 = "\nwith open('{{ filename }}', 'w') as f:\n    f.write('{{ text }}')\n"
    var_3 = 'utf-8'
    var_4 = 'filename'
    var_5 = 'text'
    var_6 = 'special.txt'
    var_7 = 'Special chars: !@#$%^&*()'
    var_8 = {var_4: var_6, var_5: var_7}

def test_case_0():
    var_0 = 'Test run_script_with_context runs from specified working directory.'
    var_1 = 'scripts'
    var_2 = 'work'
    var_3 = 'test_script.py'
    var_4 = "\nimport os\nwith open('output.txt', 'w') as f:\n    f.write(os.getcwd())\n"
    var_5 = 'utf-8'
    var_6 = {}
    var_7 = 'output.txt'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'pre_prompt.py'
    var_8 = 'print("test")'
    var_9 = len(var_3)
    assert var_9 == 2
    var_10 = 'pre_prompt.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_3)
    assert var_12 == 2
    var_13 = 'post_gen_project'
    var_14 = 'invalid_hook.sh'
    var_15 = '#!/bin/bash\necho "invalid"'
    var_16 = 'invalid_hook'
    var_17 = 'post_gen_project.py'
    var_18 = 'print("post")'
    var_19 = len(var_3)
    assert var_19 == 1



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes script after Jinja rendering.'
    var_1 = 'test_script.py'
    var_2 = "# Context: {{ cookiecutter.project_name }}\nprint('{{ cookiecutter.project_name }}')"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = 0

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context handles undefined Jinja variables.'
    var_1 = 'test_script.py'
    var_2 = '# {{ undefined_var }}'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.run_script_with_context(var_0, var_1, var_8)

def test_case_0():
    var_0 = 'Test run_script_with_context preserves file extension.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.name }}'"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = 0
    var_11 = '.sh'

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = 'test_script.py'
    var_2 = '\n# Project: {{ project_name }}\n# Author: {{ author }}\n# Version: {{ version }}\n'
    var_3 = 'utf-8'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'version'
    var_7 = 'MyProject'
    var_8 = 'John Doe'
    var_9 = '1.0.0'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context passes correct cwd to run_script.'
    var_1 = 'test_script.py'
    var_2 = "print('test')"
    var_3 = 'utf-8'
    var_4 = 'custom_dir'
    var_5 = {}
    var_6 = 'cookiecutter.hooks.run_script'
    var_7 = 0



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test run_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'hook_script.py'
    var_9 = 'print("hook executed")'
    var_10 = 'cookiecutter.hooks.find_hook'
    var_11 = 'cookiecutter.hooks.run_script_with_context'

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_hook when no scripts are found.'
    var_1 = 'pre_gen_project'
    var_2 = '/some/path'
    var_3 = {}
    var_4 = 'cookiecutter.hooks.find_hook'
    var_5 = 'cookiecutter.hooks.run_script_with_context'
    var_6 = module_0.run_hook(var_1, var_2, var_3)

def test_case_0():
    var_0 = 'Test run_hook with multiple hook scripts.'
    var_1 = 'post_gen_project'
    var_2 = 'project'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'script1.py'
    var_7 = 'script2.sh'
    var_8 = 'cookiecutter.hooks.find_hook'
    var_9 = 'cookiecutter.hooks.run_script_with_context'
    var_10 = 0
    var_11 = 1



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ cookiecutter.project_name }}')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0
    var_11 = 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context handles undefined variables gracefully.'
    var_1 = 'test_script.py'
    var_2 = "print('{{ undefined_var }}')\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = module_0.run_script_with_context(var_0, var_1, var_9)

def test_case_0():
    var_0 = 'Test run_script_with_context with bash script.'
    var_1 = 'test_script.sh'
    var_2 = "#!/bin/bash\necho '{{ cookiecutter.name }}'\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja expressions.'
    var_1 = 'test_script.py'
    var_2 = "{% if cookiecutter.use_feature %}\nprint('Feature enabled')\n{% else %}\nprint('Feature disabled')\n{% endif %}\n"
    var_3 = 'utf-8'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'use_feature'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 0



# Parsed testcases at query #33
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function with various scenarios.'
    var_1 = 'pre_prompt'
    var_2 = 'hooks'
    var_3 = module_0.find_hook(var_1, var_2)
    assert var_3 is None
    var_4 = 'pre_prompt.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'pre_prompt.py'
    var_8 = 'print("test")'
    var_9 = len(var_3)
    assert var_9 == 2
    var_10 = 'pre_prompt.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_3)
    assert var_12 == 2
    var_13 = 'post_gen_project'
    var_14 = 'unsupported_hook.sh'
    var_15 = '#!/bin/bash'
    var_16 = 'unsupported_hook'
    var_17 = 'post_gen_project.py'
    var_18 = 'print("post gen")'
    var_19 = len(var_3)
    assert var_19 == 1
    var_20 = 'non_existent_hooks'
    var_21 = module_0.find_hook(var_1, var_20)
    assert var_21 is None



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context executes a script with rendered context.'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    var_2 = 'test_script.sh'
    var_3 = 'work'
    var_4 = {}
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}

def test_case_0():
    var_0 = 'Test run_script_with_context with a Python script.'
    var_1 = "print('{{ cookiecutter.name }}')"
    var_2 = 'test_script.py'
    var_3 = 'work'
    var_4 = []
    var_5 = 'cookiecutter.hooks.run_script'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = len(var_4)
    assert var_11 == 1

import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test run_script_with_context raises UndefinedError for undefined variables.'
    var_1 = "echo '{{ undefined_var }}'"
    var_2 = 'test_script.sh'
    var_3 = 'work'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.run_script_with_context(var_0, var_2, var_8)

def test_case_0():
    var_0 = 'Test run_script_with_context with multiple context variables.'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.name }}-{{ cookiecutter.version }}'"
    var_2 = 'test_script.sh'
    var_3 = 'work'
    var_4 = 'cookiecutter.hooks.run_script'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'version'
    var_8 = 'myproject'
    var_9 = '1.0.0'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}



# Parsed testcases at query #35
#--------------------------


import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = 'Test find_hook function.'
    var_1 = 'pre_gen_project'
    var_2 = module_0.find_hook(var_1)
    assert var_2 is None
    var_3 = 'hooks'
    var_4 = 'pre_gen_project.sh'
    var_5 = '#!/bin/bash\necho "test"'
    var_6 = len(var_2)
    assert var_6 == 1
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("test")'
    var_9 = len(var_2)
    assert var_9 == 2
    var_10 = 'pre_gen_project.sh~'
    var_11 = '#!/bin/bash\necho "backup"'
    var_12 = len(var_2)
    assert var_12 == 2
    var_13 = 'post_gen_project'
    var_14 = 'post_gen_project.sh'
    var_15 = '#!/bin/bash\necho "post"'
    var_16 = len(var_2)
    assert var_16 == 1
    var_17 = 'unsupported_hook.sh'
    var_18 = '#!/bin/bash'
    var_19 = 'unsupported_hook'
    var_20 = '#!/bin/bash\necho "no ext"'
    var_21 = 'pre_prompt.sh'
    var_22 = []
    var_23 = var_2 or var_22
    var_24 = [Path(p) for p in var_23]



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test run_script_with_context renders and executes a script with context.'
    var_1 = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = None
    var_11 = []
    var_12 = 'cookiecutter.hooks.run_script'
    var_13 = len(var_11)
    assert var_13 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with Python file extension.'
    var_1 = "print('{{ cookiecutter.project_slug }}')"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'cookiecutter'
    var_6 = 'project_slug'
    var_7 = 'my_slug'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = []
    var_11 = 'cookiecutter.hooks.run_script'
    var_12 = len(var_10)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test run_script_with_context with undefined Jinja variable.'
    var_1 = "#!/bin/bash\necho '{{ undefined_var }}'"
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'Test run_script_with_context with complex Jinja template.'
    var_1 = '#!/bin/bash\n{% if cookiecutter.use_feature %}\necho "Feature enabled: {{ cookiecutter.feature_name }}"\n{% endif %}\n'
    var_2 = 'test_script.sh'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'cookiecutter'
    var_6 = 'use_feature'
    var_7 = 'feature_name'
    var_8 = True
    var_9 = 'awesome_feature'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = []
    var_13 = 'cookiecutter.hooks.run_script'
    var_14 = len(var_12)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test that temporary files are created with correct extension.'
    var_1 = "#!/usr/bin/env python\nprint('{{ test }}')"
    var_2 = 'test_script.py'
    var_3 = 'utf-8'
    var_4 = 'work_dir'
    var_5 = 'test'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = []
    var_9 = 'cookiecutter.hooks.run_script'
    var_10 = len(var_8)
    assert var_10 == 1



